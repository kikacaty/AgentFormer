import os
import numpy as np
import argparse
from data.nuscenes_pred_split import get_nuscenes_pred_split
from data.ethucy_split import get_ethucy_split
from utils.utils import print_log, AverageMeter, isfile, print_log, AverageMeter, isfile, isfolder, find_unique_common_from_lists, load_list_from_folder, load_txt_file

import cv2
import glob
from data.map import GeometricMap

from utils.attack_utils.constraint import DynamicModelOpt
from utils.config import AdvConfig

import pickle

from pdb import set_trace as st
import matplotlib.pyplot as plt

# import dill
import wandb

""" Metrics """

def compute_ADE(pred_arr, gt_arr, PI=None):
    if PI is None:
        PI = np.ones(pred_arr.shape[0])
    ade = 0.0
    for pred, gt, sens in zip(pred_arr, gt_arr, PI):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist.mean(axis=-1)                       # samples
        ade += dist.min(axis=0) * sens                  # (1, )
    ade /= len(pred_arr)
    return ade


def compute_FDE(pred_arr, gt_arr, PI=None):
    if PI is None:
        PI = np.ones(pred_arr.shape[0])
    fde = 0.0
    for pred, gt, sens in zip(pred_arr, gt_arr, PI):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist[..., -1]                            # samples 
        fde += dist.min(axis=0) * sens                         # (1, )
    fde /= len(pred_arr)
    return fde

def compute_MR(pred_arr, gt_arr, tolerance=2, PI=None):
    if PI is None:
        PI = np.ones(pred_arr.shape[0])
    mr = 0.0
    for pred, gt, sens in zip(pred_arr, gt_arr, PI):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        cur_mr = (dist > tolerance).sum(axis=-1)                            # samples 
        mr += cur_mr.min(axis=0) * sens                         # (1, )
    mr /= len(pred_arr)
    return mr/12.

def compute_violation(past_arr, gt_past_arr, device = None, cfg = None):
    dynamic_model = DynamicModelOpt(past_arr, device=device, cfg = cfg)
    k_vio, dds_vio = dynamic_model.get_violations()
    st()

    return k_vio


# def compute_ORR(pred_arr, vis_map):
#     mr = 0.0
#     for pred in pred_arr:
#         map_points = vis_map.to_map_points(pred).round().astype(int)     # samples x frames x 3
#         orig_shape = map_points.shape[:-1]
#         map_points = map_points.reshape(-1,2)
#         pixels = np.array([vis_map.data.transpose(1,2,0)[x[0],x[1]] for x in map_points]).reshape([*orig_shape,3])
#         cur_rates = (pixels == [255, 240, 243]).sum(1)[:,0]         # samples 
#         mr += cur_rates.mean(axis=0)                         # (1, )
#     mr /= len(pred_arr)
#     return mr/12.

def compute_ORR(pred_arr, scene_map, PI=None):
    if PI is None:
        PI = np.ones(pred_arr.shape[0])
    mr = 0.0
    for pred, sens in zip(pred_arr, PI):
        map_points = scene_map.to_map_points(pred).round().astype(int)     # samples x frames x 3
        orig_shape = map_points.shape[:-1]
        map_points = map_points.reshape(-1,2)
        pixels = np.array([scene_map.data.transpose(1,2,0)[x[0],x[1],0] for x in map_points]).reshape([*orig_shape])
        cur_rates = (pixels != 255).sum(1)               # samples 
        mr += cur_rates.mean(axis=0) * sens                         # (1, )

    mr /= len(pred_arr)
    return mr/12.


def align_gt(pred, gt):
    frame_from_data = pred[0, :, 0].astype('int64').tolist()
    frame_from_gt = gt[:, 0].astype('int64').tolist()
    common_frames, index_list1, index_list2 = find_unique_common_from_lists(frame_from_gt, frame_from_data)
    assert len(common_frames) == len(frame_from_data)
    gt_new = gt[index_list1, 2:]
    pred_new = pred[:, index_list2, 2:]
    return pred_new, gt_new


def evaluate(results_dir, dataset='nuscenes_pred', data='test', exclude_adv=False, device=None, dump=False):
    dataset = dataset.lower()
    results_dir = results_dir
    
    if dataset == 'nuscenes_pred':   # nuscenes
        data_root = f'datasets/nuscenes_pred'
        gt_dir = f'{data_root}/label/{data}'
        seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
        seq_eval = locals()[f'seq_{data}']
    else:                            # ETH/UCY
        gt_dir = f'datasets/eth_ucy/{dataset}'
        seq_train, seq_val, seq_test = get_ethucy_split(dataset)
        seq_eval = locals()[f'seq_{data}']

    stats_func = {
        'ADE': compute_ADE,
        'FDE': compute_FDE,
        'MissRate': compute_MR,
        'OffRoadRate': compute_ORR,
        # 'Violation': compute_violation,
    }

    eval_cfg = AdvConfig('adv_opt')

    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    seq_list, num_seq = load_list_from_folder(gt_dir)
    # print('\n\nnumber of sequences to evaluate is %d' % len(seq_eval))

    d_stats_file = 'results/nuscenes_5sample_agentformer/results/epoch_0035/test/pgd_step_fix_init_dds_0.01_dk_0.05/dist.pkl'
    with open(d_stats_file, 'rb') as reader:
        d_stats = pickle.load(reader)

    metric_map = {}

    for seq_name in seq_eval:

        metric_map[seq_name] = {x: AverageMeter() for x in stats_func.keys()}

        # load GT raw data
        map_vis_file = f'{data_root}/map_0.1/vis_{seq_name}.png'
        map_file = f'{data_root}/map_0.1/{seq_name}.png'
        map_meta_file = f'{data_root}/map_0.1/meta_{seq_name}.txt'
        scene_vis_map = np.transpose(cv2.cvtColor(cv2.imread(map_vis_file), cv2.COLOR_BGR2RGB), (2, 0, 1))
        scene_map = np.transpose(cv2.imread(map_file), (2, 0, 1))
        meta = np.loadtxt(map_meta_file)
        map_origin = meta[:2]
        map_scale = scale = meta[2]
        homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
        scene_vis_map = GeometricMap(scene_vis_map, homography, map_origin)
        scene_map = GeometricMap(scene_map, homography, map_origin)

        gt_data, _ = load_txt_file(os.path.join(gt_dir, seq_name+'.txt'))
        gt_raw = []
        for line_data in gt_data:
            line_data = np.array([line_data.split(' ')])[:, [0, 1, 13, 15]][0].astype('float32')
            if line_data[1] == -1: continue
            gt_raw.append(line_data)
        gt_raw = np.stack(gt_raw)

        data_filelist, _ = load_list_from_folder(os.path.join(results_dir, seq_name))    
            
        for data_file in data_filelist:      # each example e.g., seq_0001 - frame_000009
            
            if exclude_adv:
                d_id = seq_name+'/'+data_file.split('/')[-1]
                dist = d_stats[d_id].min()
                # if dist > 50: continue
            
            # for reconsutrction or deterministic
            if isfile(data_file):
                all_traj = np.loadtxt(data_file, delimiter=' ', dtype='float32')        # (frames x agents) x 4
                all_traj = np.expand_dims(all_traj, axis=0)                             # 1 x (frames x agents) x 4
            # for stochastic with multiple samples
            elif isfolder(data_file):
                sample_list, _ = load_list_from_folder(data_file)
                sample_all = []
                for sample in sample_list:
                    sample = np.loadtxt(sample, delimiter=' ', dtype='float32')        # (frames x agents) x 4
                    sample_all.append(sample)
                all_traj = np.stack(sample_all, axis=0)                                # samples x (framex x agents) x 4
            else:
                assert False, 'error'

            # convert raw data to our format for evaluation
            id_list, id_indices = np.unique(all_traj[:, :, 1], return_index=True)
            frame_list = np.unique(all_traj[:, :, 0])

            agent_traj = []
            gt_traj = []
            for idx in id_list:
                if exclude_adv and idx == all_traj[:, :, 1].flatten()[id_indices.min()]: continue
                # GT traj
                gt_idx = gt_raw[gt_raw[:, 1] == idx]                          # frames x 4
                # predicted traj
                ind = np.unique(np.where(all_traj[:, :, 1] == idx)[1].tolist())
                pred_idx = all_traj[:, ind, :]                                # sample x frames x 4
                # filter data
                pred_idx, gt_idx = align_gt(pred_idx, gt_idx)
                # append
                agent_traj.append(pred_idx)
                gt_traj.append(gt_idx)
            
            agent_traj = np.array(agent_traj)
            gt_traj = np.array(gt_traj)

            """compute stats"""
            for stats_name, meter in stats_meter.items():
                func = stats_func[stats_name]
                if stats_name == 'OffRoadRate':
                    value = func(agent_traj, scene_map)
                elif stats_name == 'Violation':
                    continue
                else:
                    value = func(agent_traj, gt_traj)
                meter.update(value, n=len(agent_traj))
                metric_map[seq_name][stats_name].update(value, n=len(agent_traj))

            # stats_str = ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in stats_meter.items()])
            # print(f'evaluating seq {seq_name:s}, forecasting frame {int(frame_list[0]):06d} to {int(frame_list[-1]):06d} {stats_str}')
    if dump:
        data_file_path = os.path.join(results_dir,f"scene_metrics.dill")
        # dill.dump(metric_map, open(data_file_path, mode='wb'))

    return stats_meter

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nuscenes_pred')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--data', default='test')
    parser.add_argument('--log_file', default=None)
    parser.add_argument('--wandb', action='store_true', default=False)

    args = parser.parse_args()

    dataset = args.dataset.lower()
    results_dir = args.results_dir
    
    if dataset == 'nuscenes_pred':   # nuscenes
        data_root = f'datasets/nuscenes_pred'
        gt_dir = f'{data_root}/label/{args.data}'
        seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
        seq_eval = globals()[f'seq_{args.data}']
    else:                            # ETH/UCY
        gt_dir = f'datasets/eth_ucy/{args.dataset}'
        seq_train, seq_val, seq_test = get_ethucy_split(args.dataset)
        seq_eval = globals()[f'seq_{args.data}']

    if args.log_file is None:
        log_file = os.path.join(results_dir, 'log_eval.txt')
    else:
        log_file = args.log_file
    log_file = open(log_file, 'a+')
    print_log('loading results from %s' % results_dir, log_file)
    print_log('loading GT from %s' % gt_dir, log_file)

    if args.wandb:
        wandb.init(project="robust_pred", entity="yulongc")

        exp_name_wandb = f'DLOW_{args.log_file}'
        wandb.run.name = exp_name_wandb
        wandb.run.save()

    stats_func = {
        'ADE': compute_ADE,
        'FDE': compute_FDE,
        'MissRate': compute_MR,
        'OffRoadRate': compute_ORR,
    }

    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    seq_list, num_seq = load_list_from_folder(gt_dir)
    print_log('\n\nnumber of sequences to evaluate is %d' % len(seq_eval), log_file)
    for seq_name in seq_eval:
        # load GT raw data
        map_vis_file = f'{data_root}/map_0.1/vis_{seq_name}.png'
        map_file = f'{data_root}/map_0.1/{seq_name}.png'
        map_meta_file = f'{data_root}/map_0.1/meta_{seq_name}.txt'
        scene_vis_map = np.transpose(cv2.cvtColor(cv2.imread(map_vis_file), cv2.COLOR_BGR2RGB), (2, 0, 1))
        scene_map = np.transpose(cv2.imread(map_file), (2, 0, 1))
        meta = np.loadtxt(map_meta_file)
        map_origin = meta[:2]
        map_scale = scale = meta[2]
        homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
        scene_vis_map = GeometricMap(scene_vis_map, homography, map_origin)
        scene_map = GeometricMap(scene_map, homography, map_origin)

        gt_data, _ = load_txt_file(os.path.join(gt_dir, seq_name+'.txt'))
        gt_raw = []
        for line_data in gt_data:
            line_data = np.array([line_data.split(' ')])[:, [0, 1, 13, 15]][0].astype('float32')
            if line_data[1] == -1: continue
            gt_raw.append(line_data)
        gt_raw = np.stack(gt_raw)

        data_filelist, _ = load_list_from_folder(os.path.join(results_dir, seq_name))    
            
        for data_file in data_filelist:      # each example e.g., seq_0001 - frame_000009
            # for reconsutrction or deterministic
            if isfile(data_file):
                all_traj = np.loadtxt(data_file, delimiter=' ', dtype='float32')        # (frames x agents) x 4
                all_traj = np.expand_dims(all_traj, axis=0)                             # 1 x (frames x agents) x 4
            # for stochastic with multiple samples
            elif isfolder(data_file):
                sample_list, _ = load_list_from_folder(data_file)
                sample_all = []
                for sample in sample_list:
                    sample = np.loadtxt(sample, delimiter=' ', dtype='float32')        # (frames x agents) x 4
                    sample_all.append(sample)
                all_traj = np.stack(sample_all, axis=0)                                # samples x (framex x agents) x 4
            else:
                assert False, 'error'

            # convert raw data to our format for evaluation
            id_list = np.unique(all_traj[:, :, 1])
            frame_list = np.unique(all_traj[:, :, 0])
            agent_traj = []
            gt_traj = []
            for idx in id_list:
                # GT traj
                gt_idx = gt_raw[gt_raw[:, 1] == idx]                          # frames x 4
                # predicted traj
                ind = np.unique(np.where(all_traj[:, :, 1] == idx)[1].tolist())
                pred_idx = all_traj[:, ind, :]                                # sample x frames x 4
                # filter data
                pred_idx, gt_idx = align_gt(pred_idx, gt_idx)
                # append
                agent_traj.append(pred_idx)
                gt_traj.append(gt_idx)

            agent_traj = np.array(agent_traj)
            gt_traj = np.array(gt_traj)

            """compute stats"""
            for stats_name, meter in stats_meter.items():
                func = stats_func[stats_name]
                if stats_name == 'OffRoadRate':
                    value = func(agent_traj, scene_map)
                else:
                    value = func(agent_traj, gt_traj)
                meter.update(value, n=len(agent_traj))

            stats_str = ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in stats_meter.items()])
            print_log(f'evaluating seq {seq_name:s}, forecasting frame {int(frame_list[0]):06d} to {int(frame_list[-1]):06d} {stats_str}', log_file)

            if args.wandb:
                wandb_log = {}
                for x, y in stats_meter.items():
                    wandb_log[x] = y.avg
                wandb.log(wandb_log)

    print_log('-' * 30 + ' STATS ' + '-' * 30, log_file)
    for name, meter in stats_meter.items():
        print_log(f'{meter.count} {name}: {meter.avg:.4f}', log_file)
    print_log('-' * 67, log_file)
    log_file.close()
