from turtle import update
import numpy as np
import argparse
import os
import sys
import subprocess
import shutil

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config, AdvConfig
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing

from eval import AverageMeter, compute_ADE, compute_FDE, compute_MR, compute_ORR

from tqdm import tqdm

from matplotlib import collections  as mc
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  
from matplotlib.lines import Line2D
from pdb import set_trace as st

from utils.attack_utils.constraint import DynamicModel, DynamicStats
from utils.attack_utils.attack import Attacker

import pickle

import wandb

import re

from eval import evaluate

def get_model_prediction(data, sample_k, model):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D

""" Attack """
def cal_dist(gt_motion):
    diff = gt_motion[1:,:,:] - gt_motion[0,:,:].tile(gt_motion.size(0)-1,1,1)
    avg_dist = diff.norm(dim=2).mean(axis=1)
    return avg_dist
    


def get_adv_model_prediction(data, sample_k, model, adv_cfg=None):
    model.set_data(data)

    attacker = Attacker(model, adv_cfg)
    attacker.perturb(data)

    return attacker.recon_motion_list, attacker.sample_motion_list

def save_prediction(pred, data, suffix, save_dir, cfg=None):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']

    for i in range(len(valid_id)):    # number of agents
        identity = valid_id[i]
        if pred_mask is not None and pred_mask[i] != 1.0:
            continue

        """future frames"""
        for j in range(cfg.future_frames):
            cur_data = fut_data[j]
            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                data = most_recent_data.copy()
                data[0] = frame + j + 1
            data[[13, 15]] = pred[i, j].cpu().numpy()   # [13, 15] corresponds to 2D pos
            most_recent_data = data.copy()
            pred_arr.append(data)
        pred_num += 1

    if len(pred_arr) > 0:
        pred_arr = np.vstack(pred_arr)
        indices = [0, 1, 13, 15]            # frame, ID, x, z (remove y which is the height)
        pred_arr = pred_arr[:, indices]
        # save results
        fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
        mkdir_if_missing(fname)
        np.savetxt(fname, pred_arr, fmt="%.3f")
    return pred_num

def save_past(pre_motion, data, suffix, save_dir, cfg=None):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame, valid_id, pred_mask = data['pre_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']

    pre_motion = pre_motion.clone().transpose(0,1)

    for i in range(len(valid_id)):    # number of agents
        identity = valid_id[i]
        if pred_mask is not None and pred_mask[i] != 1.0:
            continue

        """pre frames"""
        for j in range(cfg.past_frames):
            cur_data = fut_data[j]
            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                data = most_recent_data.copy()
                data[0] = frame + j + 1
            data[[13, 15]] = pre_motion[i, j].cpu().numpy()   # [13, 15] corresponds to 2D pos
            most_recent_data = data.copy()
            pred_arr.append(data)
        pred_num += 1

    if len(pred_arr) > 0:
        pred_arr = np.vstack(pred_arr)
        indices = [0, 1, 13, 15]            # frame, ID, x, z (remove y which is the height)
        pred_arr = pred_arr[:, indices]
        # save results
        fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
        mkdir_if_missing(fname)
        np.savetxt(fname, pred_arr, fmt="%.3f")
    return pred_num

def attack_model(model, generator, save_dir, cfg, args):
    total_num_pred = 0
    total_num_frame = 0
    dist_stats = torch.zeros(6)
    dist_cat = torch.tensor([5,6,7,8,9,10],dtype=torch.float)

    pbar = tqdm(total=generator.num_total_samples)

    eval_scenes = {}
    dist_label = {}

    device = model.device
    adv_cfg = cfg.adv_cfg

    stats_func = {
        'ADE': compute_ADE,
        'FDE': compute_FDE,
        'MissRate': compute_MR,
        'OffRoadRate': compute_ORR,
    }

    adv_stats_meter = {x: AverageMeter() for x in stats_func.keys()}
    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    while not generator.is_epoch_end():
        data = generator()
        skip = False
        if data is None:
            continue
        if data['pred_mask'].sum() < 2:
            skip = True

        # result_log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
        # with open(result_log_file,'r') as f:
        #     result_log = f.read()
            
        #     results = re.findall(r"forecasting frame %06d .+ ADE: (\d+\.\d+)"%(data['frame']+1), result_log)
        #     results = float(results[0])
            # if results > 0.1:
            #     continue 
        
        # seq_name, frame = data['seq'], int(data['frame'])

        # ================= Start filtering input traces for attack =================

        gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
        gt_pre_motion_3D = torch.stack(data['pre_motion_3D'], dim=0).to(device) * cfg.traj_scale
        
        # avg_dist = cal_dist(gt_motion_3D)
        
        # avg_movement = (gt_motion_3D[:,1:,:] - gt_motion_3D[:,:-1,:]).norm(dim=2).sum(axis=1).mean()
        # nade = results/avg_movement
        
        # print(results, nade.item())
        # if nade > 0.2:
        #     continue
        # dist_stats += (dist_cat > avg_movement.max().cpu())
        # print(dist_stats,avg_movement)

        # skip short adv history scenarios
        if 0 in data['pre_motion_mask'][0]:
            skip = True

        # skip low speed scenarios
        adv_pre = data['pre_motion_3D'][0]* cfg.traj_scale / 0.5
        adv_pre_s = (adv_pre[1:] - adv_pre[:-1]) 
        adv_fut = data['fut_motion_3D'][0]* cfg.traj_scale / 0.5
        adv_fute_s = (adv_fut[1:] - adv_fut[:-1]) 
        if adv_pre_s.norm(dim=1).min() < 1 or adv_fute_s.norm(dim=1).min() < 1:
            skip = True
        
        if len(adv_cfg.ttl_frame) != 0:
            if data['seq'] in eval_scenes.keys():
                skip = True
            if skip:
                continue
            if total_num_frame not in [6,8,22,30,39,57,62,67,70,98,100]:
                eval_scenes[data['seq']] = True
                total_num_frame += 1 
                continue

        # ================= End filtering input traces for attack =================

        if skip:
            if args.clean_results:
                for idx in range(len(adv_cfg.iters)):
                    adv_sample_dir = os.path.join(save_dir, f'samples_adv/step_{adv_cfg.iters[idx]}')
                    fname = f"{adv_sample_dir}/{data['seq']}/frame_{int(data['frame']):06d}"
                    shutil.rmtree(fname, ignore_errors=True)
                    baseline_sample_dir = os.path.join(save_dir, f'samples_baseline')
                    fname = f"{baseline_sample_dir}/{data['seq']}/frame_{int(data['frame']):06d}"
                    shutil.rmtree(fname, ignore_errors=True)
            continue
        if args.clean_results:
            continue


        diff = np.array([((data['pre_motion_3D'][0]-x)* cfg.traj_scale).norm(dim=-1).min() for x in data['pre_motion_3D'][1:]])
        dist_label[f"{data['seq']}/frame_{int(data['frame']):06d}"] = diff


        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k, model)

        recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

        """save samples"""
        recon_dir = os.path.join(save_dir, 'recon_baseline'); mkdir_if_missing(recon_dir)
        sample_dir = os.path.join(save_dir, 'samples_baseline'); mkdir_if_missing(sample_dir)
        past_dir = os.path.join(save_dir, 'past_baseline'); mkdir_if_missing(past_dir)
        gt_dir = os.path.join(save_dir, 'gt'); mkdir_if_missing(gt_dir)
        for i in range(sample_motion_3D.shape[0]):
            save_prediction(sample_motion_3D[i], data, f'/sample_{i:03d}', sample_dir, cfg=cfg)
        save_prediction(recon_motion_3D, data, '', recon_dir, cfg=cfg)        # save recon
        num_pred = save_prediction(gt_motion_3D, data, '', gt_dir, cfg=cfg)   # save gt
        save_past(model.data['pre_motion'], data, '', past_dir, cfg=cfg)


        # generate adv
        adv_recon_motion_list, adv_sample_motion_list = get_adv_model_prediction(data, cfg.sample_k, model, adv_cfg=adv_cfg)
        
        # adv_agent_pre_path = np.array([(model.data['pre_vel'][:i,0,:].sum(axis=0) + model.data['pre_motion'][0,0,:]).cpu().numpy() for i in range(len(model.data['pre_vel'])+1)])
        adv_agent_pre_path = model.data['pre_motion'][:,0,:].cpu().numpy()
        adv_agent_pre_path = adv_agent_pre_path * cfg.traj_scale

        """save samples"""
        for idx, (adv_recon_motion, adv_sample_motion) in enumerate(zip(adv_recon_motion_list, adv_sample_motion_list)):
            adv_recon_motion_3D, adv_sample_motion_3D = adv_recon_motion * cfg.traj_scale, adv_sample_motion * cfg.traj_scale

            adv_recon_dir = os.path.join(save_dir, f'recon_adv/step_{adv_cfg.iters[idx]}'); mkdir_if_missing(adv_recon_dir)
            adv_sample_dir = os.path.join(save_dir, f'samples_adv/step_{adv_cfg.iters[idx]}'); mkdir_if_missing(adv_sample_dir)
            adv_past_dir = os.path.join(save_dir, 'past_adv'); mkdir_if_missing(adv_past_dir)

            for i in range(adv_sample_motion_3D.shape[0]):
                save_prediction(adv_sample_motion_3D[i], data, f'/sample_{i:03d}', adv_sample_dir, cfg=cfg)
            save_prediction(adv_recon_motion_3D, data, '', adv_recon_dir, cfg=cfg)        # save adv recon
            save_past(model.data['pre_motion'], data, '', adv_past_dir, cfg=cfg)


        # eval
        adv_list = [0]
        scene_vis_map = data['scene_map_vis']
        agent_traj = []
        adv_agent_traj = []
        gt_traj = []
        for idx in range(adv_sample_motion_3D.shape[1]):

            if idx in adv_list: continue

            # adv_pred_idx = adv_sample_motion_3D[0,idx,:].unsqueeze(0).cpu().numpy()
            # pred_idx = sample_motion_3D[0,idx,:].unsqueeze(0).cpu().numpy()
            adv_pred_idx = adv_sample_motion_3D[:,idx,:].cpu().numpy()
            pred_idx = sample_motion_3D[:,idx,:].cpu().numpy()
            gt_idx = gt_motion_3D[idx,:].cpu().numpy()

            adv_agent_traj.append(adv_pred_idx)
            agent_traj.append(pred_idx)
            gt_traj.append(gt_idx)

        adv_agent_traj = np.array(adv_agent_traj)
        agent_traj = np.array(agent_traj)
        gt_traj = np.array(gt_traj)

        """compute stats"""
        for stats_name, meter in stats_meter.items():
            func = stats_func[stats_name]
            if stats_name == 'OffRoadRate':
                value = func(agent_traj, scene_vis_map)
            else:
                value = func(agent_traj, gt_traj)
            meter.update(value, n=len(agent_traj))
        
        for stats_name, meter in adv_stats_meter.items():
            func = stats_func[stats_name]
            if stats_name == 'OffRoadRate':
                value = func(adv_agent_traj, scene_vis_map)
            else:
                value = func(adv_agent_traj, gt_traj)
            meter.update(value, n=len(adv_agent_traj))

        stats_str = ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in stats_meter.items()])
        adv_stats_str = ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in adv_stats_meter.items()])
        print(f'evaluating seq {data["seq"]:s}\norig {stats_str}')
        print(f'adv {adv_stats_str}')

        wandb_log = {}
        for x, y in adv_stats_meter.items():
            wandb_log[x] = y.avg
        # wandb.log(wandb_log)
        


        # visualization
        if args.vis:

            # map visualization
            # data['pre_motion_3D'] = model.data['pre_motion'].transpose(1,0).detach().cpu()
            # data['adv_fut_motion_3D'] = adv_sample_motion[0].detach().cpu()
            # data['scene_map_vis'].visualize_data(data)

            vis_dir = os.path.join(save_dir, 'vis'); mkdir_if_missing(vis_dir)

            fig = plt.figure(figsize=[10,10])
            ax = fig.add_subplot(111)

            line_w = 2
            marker_s = 3

            pre_lines = []
            fut_lines = []
            pred_lines = []
            adv_pred_lines = []
            

            # adv agent
            idx = 0

            fut_path = gt_motion_3D[idx].cpu().data.numpy()
            pre_path = gt_pre_motion_3D[idx].cpu().data.numpy()
            pred_path = sample_motion_3D[0][idx].cpu().data.numpy()
            adv_pred_path = adv_sample_motion_3D[0][idx].cpu().data.numpy()

            adv_path = np.concatenate([pre_path,adv_pred_path],axis=0)

            fut_path = np.insert(fut_path, 0,pre_path[-1],axis=0)
            pred_path = np.insert(pred_path, 0,pre_path[-1],axis=0)
            adv_pred_path = np.insert(adv_pred_path, 0,adv_agent_pre_path[-1],axis=0)

            c = ['C0','C1','C2','C3','C4']

            pre_lines.append((ax.plot(pre_path[:,0], pre_path[:,1], color=c[0],marker='^', linewidth=line_w, markersize=marker_s,alpha=0.5)[0],pre_path))
            adv_agent_pre_line = ax.plot(adv_agent_pre_path[:,0], adv_agent_pre_path[:,1], color=c[4],marker='^', markersize=marker_s,alpha=0.5)[0]
            fut_lines.append((ax.plot(fut_path[:,0], fut_path[:,1], color=c[1],marker='^', linewidth=line_w, markersize=marker_s,alpha=0.5)[0],fut_path))
            pred_lines.append((ax.plot(pred_path[:,0], pred_path[:,1], color=c[2],marker='^', linewidth=line_w, markersize=marker_s,alpha=0.5)[0],pred_path))
            adv_pred_lines.append((ax.plot(adv_pred_path[:,0], adv_pred_path[:,1], color=c[3],marker='^', linewidth=line_w, markersize=marker_s,alpha=0.5)[0],adv_pred_path))

            # other agents
            for idx in range(len(data['pre_motion_3D'])-1):
                
                idx += 1

                fut_path = gt_motion_3D[idx].cpu().data.numpy()
                pre_path = gt_pre_motion_3D[idx].cpu().data.numpy()
                pred_path = sample_motion_3D[0][idx].cpu().data.numpy()
                adv_pred_path = adv_sample_motion_3D[0][idx].cpu().data.numpy()

                fut_path = np.insert(fut_path, 0,pre_path[-1],axis=0)
                pred_path = np.insert(pred_path, 0,pre_path[-1],axis=0)
                adv_pred_path = np.insert(adv_pred_path, 0,pre_path[-1],axis=0)
                
                pre_lines.append((ax.plot(pre_path[:,0], pre_path[:,1], color=c[0],marker='o', linewidth=line_w, markersize=marker_s,alpha=0.5)[0],pre_path))
                fut_lines.append((ax.plot(fut_path[:,0], fut_path[:,1], color=c[1],marker='o', linewidth=line_w, markersize=marker_s,alpha=0.5)[0],fut_path))
                pred_lines.append((ax.plot(pred_path[:,0], pred_path[:,1], color=c[2],marker='o', linewidth=line_w, markersize=marker_s,alpha=0.5)[0],pred_path))
                # adv_pred_lines.append((ax.plot(adv_pred_path[:,0], adv_pred_path[:,1], color=c[3],marker='o', linewidth=line_w, markersize=marker_s,alpha=0.5)[0],adv_pred_path))



            legend_elements = [Line2D([0], [0], color=c[0], lw=2, label='pre'),
                            Line2D([0], [0], color=c[1], lw=2, label='gt_fut'),
                            Line2D([0], [0], color=c[2], lw=2, label='pred_fut'),
                            Line2D([0], [0], color=c[3], lw=2, label='adv_pred_fut'),
                            Line2D([0], [0], color=c[4], lw=2, label='adv_pre'),
                    Line2D([0], [0], color='w', markerfacecolor='black', marker='^', label='adv agent'),
                    Line2D([0], [0], color='w', markerfacecolor='black', marker='o', label='benign agent')]
            
            window = 50
            window_buff = 5
            # centerx, centery = np.mean(plt.xlim()), np.mean(plt.ylim())
            centerx, centery = np.mean(adv_path,axis=0)
            plt.xlim((centerx - window, centerx + window))
            plt.ylim((centery - window, centery + window))
            
            plt.xlim((centerx - window, centerx + window))
            plt.ylim((centery - window, centery + window))

            # x_min = np.min([gt_motion_3D.cpu().data.reshape(-1,2)[:,0].min(), gt_pre_motion_3D.cpu().data.reshape(-1,2)[:,0].min()]) - window_buff
            # x_max = np.max([gt_motion_3D.cpu().data.reshape(-1,2)[:,0].max(), gt_pre_motion_3D.cpu().data.reshape(-1,2)[:,0].max()]) + window_buff
            # y_min = np.min([gt_motion_3D.cpu().data.reshape(-1,2)[:,1].min(), gt_pre_motion_3D.cpu().data.reshape(-1,2)[:,1].min()]) - window_buff
            # y_max = np.max([gt_motion_3D.cpu().data.reshape(-1,2)[:,1].max(), gt_pre_motion_3D.cpu().data.reshape(-1,2)[:,1].max()]) + window_buff
            # plt.xlim((x_min, x_max))
            # plt.ylim((y_min, y_max))

            ax.margins(0.1)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            # ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
            ax.legend(handles=legend_elements, bbox_to_anchor=(1, -0.05), fancybox=True, ncol=4)
            # ax.legend(handles=legend_elements)
            lim = np.stack([plt.xlim(),plt.ylim()]).transpose()
            lim_pos = np.round(data['scene_map_vis'].to_map_points(lim)).astype(int)
            cropped_map = data['scene_map_vis'].data.transpose(2,1,0)[lim_pos[0,1]:lim_pos[1,1],lim_pos[0,0]:lim_pos[1,0],:]
            cropped_map = np.flip(cropped_map,0)
            ax.imshow(cropped_map, alpha=0.8, extent=[plt.xlim()[0], plt.xlim()[1], plt.ylim()[0], plt.ylim()[1]])
            
            def update(num, adv):
                num += 1
                if num <= cfg.past_frames:
                    if adv:
                        adv_agent_pre_line.set_data(adv_agent_pre_path[:num,0],adv_agent_pre_path[:num,1])
                    else:
                        adv_agent_pre_line.set_data([],[])  
                        for line, line_data in pre_lines:
                            line.set_data(line_data[:num,0],line_data[:num,1])
                    for line, line_data in pred_lines:
                        line.set_data([],[])
                    for line, line_data in adv_pred_lines:
                            line.set_data([],[])  
                    for line, line_data in fut_lines:
                        line.set_data([],[])
                    # if adv:
                    #     adv_agent_pre_line.set_data(adv_agent_pre_path[:num,0],adv_agent_pre_path[:num,1])
                    # else:
                    #     adv_agent_pre_line.set_data([],[])   
                else:
                    if adv:
                        for line, line_data in adv_pred_lines:
                            line.set_data(line_data[:num-cfg.past_frames,0],line_data[:num-cfg.past_frames,1])
                    else:
                        for line, line_data in fut_lines:
                            line.set_data(line_data[:num-cfg.past_frames,0],line_data[:num-cfg.past_frames,1])
                        for line, line_data in pred_lines:
                            line.set_data(line_data[:num-cfg.past_frames,0],line_data[:num-cfg.past_frames,1])
            
            
            line_ani = FuncAnimation(fig, update, frames=20, fargs=(False,),
                                    interval=500, blit=False)
            writer = PillowWriter(fps=2)  
            line_ani.save(os.path.join(vis_dir, "%03d_demo.gif" % total_num_frame), writer='pillow')  
            
            plt.savefig(os.path.join(vis_dir, '%03d.png' % total_num_frame))
            
            line_ani_adv = FuncAnimation(fig, update, frames=20, fargs=(True,),
                                    interval=500, blit=False)
            writer = PillowWriter(fps=2)  
            line_ani_adv.save(os.path.join(vis_dir, "%03d_adv_demo.gif" % total_num_frame), writer='pillow')  
            
            plt.savefig(os.path.join(vis_dir, "%03d_adv.png" % total_num_frame))
            plt.close('all')

        total_num_pred += num_pred
        total_num_frame += 1
        if len(adv_cfg.ttl_frame) > 0 and adv_cfg.ttl_frame[0] < total_num_frame:
            break
        eval_scenes[data['seq']] = True

        if adv_cfg.debug:
            pbar.update(generator.index-pbar.n)

    print_log(f'\n\n total_num_pred: {total_num_pred}', log)
    if cfg.dataset == 'nuscenes_pred':
        scene_num = {
            'train': 32186,
            'val': 8560,
            'test': 9041
        }
        # assert total_num_pred == scene_num[generator.split]
    outname = f'{save_dir}/dist.pkl'
    with open(outname, 'wb') as writer:
        pickle.dump(dist_label, writer)

def attack(args, cfg, adv_cfg):

    if args.epochs is None:
        epochs = [cfg.get_last_epoch()]
    else:
        epochs = [int(x) for x in args.epochs.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)

    adv_cfg.device = device

    if args.eval:
        for epoch in epochs:
            data_splits = [args.data_eval]
            for split in data_splits:  
                results = []
                save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}/{adv_cfg.exp_name}'
                eval_dir = f'{save_dir}/samples_baseline'
                stats_meter = evaluate(eval_dir, exclude_adv=adv_cfg.exclude_adv,device=device,dump=True)
                results.append(f'0\t' + '\t'.join([f'{y.avg:.4f}' for x, y in stats_meter.items()]))
                for iters in adv_cfg.iters:
                    adv_eval_dir = f'{save_dir}/samples_adv/step_{iters}'
                    stats_meter = evaluate(adv_eval_dir, exclude_adv=adv_cfg.exclude_adv, device=device,dump=True)
                    results.append(f'{iters}\t' + '\t'.join([f'{y.avg:.4f}' for x, y in stats_meter.items()]))
        print('iters\t'+'\t'.join([f'{x}' for x, y in stats_meter.items()]))
        print('\n'.join(results))

    else:
        # enable grad for pgd attack
        # torch.set_grad_enabled(False)
        global log
        log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

        for epoch in epochs:
            prepare_seed(cfg.seed)
            """ model """
            if not args.cached:
                model_id = cfg.get('model_id', 'gnnv1')
                model = model_dict[model_id](cfg)
                model.set_device(device)
                model.eval()
                if epoch > 0:
                    cp_path = cfg.model_path % epoch
                    print_log(f'loading model model from checkpoint: {cp_path}', log, display=True)
                    model_cp = torch.load(cp_path, map_location='cpu')
                    model.load_state_dict(model_cp['model_dict'], strict=False)

            """ save results and compute metrics """
            data_splits = [args.data_eval]
            results = []
            for split in data_splits:  
                generator = data_generator(cfg, log, split=split, phase='testing')
                save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}/{adv_cfg.exp_name}'; mkdir_if_missing(save_dir)
                eval_dir = f'{save_dir}/samples_baseline'
                if not args.cached:
                    attack_model(model, generator, save_dir, cfg, args)

                stats_meter = evaluate(eval_dir, exclude_adv=adv_cfg.exclude_adv,device=device)
                results.append(f'0\t' + '\t'.join([f'{y.avg:.4f}' for x, y in stats_meter.items()]))

                for iters in adv_cfg.iters:
                    adv_eval_dir = f'{save_dir}/samples_adv/step_{iters}'
                    stats_meter = evaluate(adv_eval_dir, exclude_adv=adv_cfg.exclude_adv,device=device)
                    results.append(f'{iters}\t' + '\t'.join([f'{y.avg:.4f}' for x, y in stats_meter.items()]))

                print('iters\t'+'\t'.join([f'{x}' for x, y in stats_meter.items()]))
                print('\n'.join(results))
                # remove eval folder to save disk space
                if args.cleanup:
                    shutil.rmtree(save_dir)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)

    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--clean_results', action='store_true', default=False)

    # ================= Attack args =================
    parser.add_argument('--adv_cfg', default='base')
    parser.add_argument('--sweep', action='store_true', default=False)
    parser.add_argument('--step_size_dds', type=float, default=0)
    parser.add_argument('--step_size_dk', type=float, default=0)
    parser.add_argument('--fix_t', type=int, default=0)
    parser.add_argument('--step_size', type=float, default=0)

    parser.add_argument('--debug', action='store_true', default=False)



    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
    adv_cfg = AdvConfig(args.adv_cfg)
    adv_cfg.sample_k = cfg.sample_k
    adv_cfg.traj_scale = cfg.traj_scale

    adv_cfg.debug = args.debug

    step_size_dds, step_size_dk, fix_t, step_size = args.step_size_dds, args.step_size_dk, args.fix_t, args.step_size

    if args.sweep:
        if adv_cfg.mode == 'opt':
            adv_cfg.step_size_dds = step_size_dds
            adv_cfg.step_size_dk = step_size_dk
            adv_cfg.fix_t.t_idx = fix_t
            fix_name = 'end' if adv_cfg.fix_t.t_idx == -1 else 'start'
            exp_name = f'sweeps/{adv_cfg.mode}_{fix_name}_dds_{adv_cfg.step_size_dds}_dk_{adv_cfg.step_size_dk}'

            adv_cfg.exp_name = exp_name

            cfg.adv_cfg = adv_cfg

        else:
            exp_name = f'sweeps/{adv_cfg.mode}_step_size_{adv_cfg.step_size}'
            adv_cfg.step_size = step_size
            adv_cfg.exp_name = exp_name

            cfg.adv_cfg = adv_cfg

    # adv_cfg.exp_name = exp_name

    # if not args.eval:
    #     # set up wandb
    #     wandb.init(project="robust_pred", entity="yulongc")

    #     wandb.run.name = adv_cfg.exp_name
    #     wandb.run.save()

    cfg.adv_cfg = adv_cfg

    attack(args, cfg, adv_cfg)


