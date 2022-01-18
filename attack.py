import numpy as np
import argparse
import os
import sys
import subprocess
import shutil

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing

from matplotlib import collections  as mc
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  
from matplotlib.lines import Line2D
from pdb import set_trace as st

import re


# class ped_model:
#     def __init__(self):


def get_model_prediction(data, sample_k):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D

def motion_bending_loss(fut_motion, pred_motion):
    try:
        diff = (fut_motion - pred_motion).cpu()
    except:
        st()
    loss_unweighted = diff.pow(2).sum()
    bend_weight = torch.from_numpy(np.array([(12-i) for i in range(12)]))
    loss_weighted = (diff.pow(2).sum(1) * bend_weight).sum()
    return loss_weighted

def dynamic_model():
    pass

def cal_dist(gt_motion):
    diff = gt_motion[1:,:,:] - gt_motion[0,:,:].tile(gt_motion.size(0)-1,1,1)
    avg_dist = diff.norm(dim=2).mean(axis=1)
    return avg_dist
    


def get_adv_model_prediction(data, sample_k, alpha = 1e-2, eps = 1, iterations=10):
    model.set_data(data)
    in_data = data
    if iterations == 0:
        recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)

        sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
        sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()

        return recon_motion_3D, sample_motion_3D

    ori_pre_motion = model.data['pre_vel']
    perturb_mask = torch.zeros_like(model.data['pre_vel'])
    perturb_mask[...,0] = 1
    target_id = 1
    for i in range(iterations):
        perturbed_pre_motion = model.data['pre_vel'].data
        # for key in model.data.keys():
        #     if model.data[key]
        #     model.data[key].requires_grad = False
        model.data['pre_vel'].requires_grad = True
        target_fut_motion = model.data['fut_motion'][:,1,:]

        recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)

        sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
        sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()

        target_pred_motion = sample_motion_3D[1][0]
        model.zero_grad()

        cost = motion_bending_loss(target_fut_motion, target_pred_motion).to(device)
        print(cost.item())
        cost.backward()

        adv_pre_motion = perturbed_pre_motion + alpha*perturb_mask*model.data['pre_vel'].grad.sign()
        eta = torch.clamp(adv_pre_motion - ori_pre_motion, min=-eps, max=eps)
        model.data['pre_vel'] = (ori_pre_motion + eta).detach_()
        
    
        update_pre_motion = torch.zeros_like(model.data['pre_motion'])
        update_pre_motion[1:,...] = torch.cumsum(model.data['pre_vel'],dim=0) 
        update_pre_motion += model.data['pre_motion'][0,...]
        model.data['pre_motion'] = update_pre_motion
        model.update_data(model.data, in_data)

    return recon_motion_3D, sample_motion_3D

def save_prediction(pred, data, suffix, save_dir):
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

def test_model(generator, save_dir, cfg):
    total_num_pred = 0
    total_num_frame = 0
    dist_stats = torch.zeros(6)
    dist_cat = torch.tensor([5,6,7,8,9,10],dtype=torch.float)
    while not generator.is_epoch_end():
        data = generator()
        if data is None or len(data['valid_id']) < 2:
            continue
        
        result_log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
        with open(result_log_file,'r') as f:
            result_log = f.read()
            
            results = re.findall(r"forecasting frame %06d .+ ADE: (\d+\.\d+)"%(data['frame']+1), result_log)
            results = float(results[0])
            # if results > 0.1:
            #     continue 
        
        seq_name, frame = data['seq'], data['frame']
        frame = int(frame)
        # sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
        # sys.stdout.flush()

        gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
        gt_pre_motion_3D = torch.stack(data['pre_motion_3D'], dim=0).to(device) * cfg.traj_scale
        
        avg_dist = cal_dist(gt_motion_3D)
        
        avg_movement = (gt_motion_3D[:,1:,:] - gt_motion_3D[:,:-1,:]).norm(dim=2).sum(axis=1).mean()
        nade = results/avg_movement
        
        # print(results, nade.item())
        if nade > 0.2:
            continue
        dist_stats += (dist_cat > avg_movement.max().cpu())
        print(dist_stats,avg_movement)
        
        
        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)

        recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

        # generate adv
        adv_recon_motion_3D, adv_sample_motion_3D = get_adv_model_prediction(data, cfg.sample_k, iterations=20, alpha = 1e-2, eps=1e-1)
        adv_recon_motion_3D, adv_sample_motion_3D = adv_recon_motion_3D * cfg.traj_scale, adv_sample_motion_3D * cfg.traj_scale
        
        adv_agent_pre_path = np.array([(model.data['pre_vel'][:i,0,:].sum(axis=0) + model.data['pre_motion'][0,0,:]).cpu().numpy() for i in range(len(model.data['pre_vel'])+1)])
        adv_agent_pre_path = adv_agent_pre_path * cfg.traj_scale
        
        # remove grad
        adv_recon_motion_3D = adv_recon_motion_3D.detach()
        adv_sample_motion_3D = adv_sample_motion_3D.detach()

        """save samples"""
        recon_dir = os.path.join(save_dir, 'recon_baseline'); mkdir_if_missing(recon_dir)
        sample_dir = os.path.join(save_dir, 'samples_baseline'); mkdir_if_missing(sample_dir)
        adv_recon_dir = os.path.join(save_dir, 'recon_adv'); mkdir_if_missing(recon_dir)
        adv_sample_dir = os.path.join(save_dir, 'samples_adv'); mkdir_if_missing(sample_dir)
        gt_dir = os.path.join(save_dir, 'gt'); mkdir_if_missing(gt_dir)
        for i in range(sample_motion_3D.shape[0]):
            save_prediction(sample_motion_3D[i], data, f'/sample_{i:03d}', sample_dir)
        save_prediction(recon_motion_3D, data, '', recon_dir)        # save recon
        for i in range(adv_sample_motion_3D.shape[0]):
            save_prediction(adv_sample_motion_3D[i], data, f'/sample_{i:03d}', adv_sample_dir)
        save_prediction(adv_recon_motion_3D, data, '', adv_recon_dir)        # save adv recon
        num_pred = save_prediction(gt_motion_3D, data, '', gt_dir)   # save gt

        # visualization
        if args.vis:
            vis_dir = os.path.join(save_dir, 'vis'); mkdir_if_missing(vis_dir)

            fig = plt.figure(figsize=[10,5])
            ax = fig.add_subplot(111)

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

            fut_path = np.insert(fut_path, 0,pre_path[-1],axis=0)
            pred_path = np.insert(pred_path, 0,pre_path[-1],axis=0)
            adv_pred_path = np.insert(adv_pred_path, 0,adv_agent_pre_path[-1],axis=0)

            # c = np.array([[0, 1, 1, 1],[1, 0, 1, 1], [0, 1, 0, 1],[0, 0, 1, 1],[1, 0, 0, 1]])
            # lines = [pre_path,adv_agent_pre_path,fut_path,pred_path,adv_pred_path]
            # lc = mc.LineCollection(lines, colors=c, linewidths=2) #, linestyle = "dashed")
            # ax.add_collection(lc)



            # ax.scatter(pre_path[:,0], pre_path[:,1], c=np.tile(c[0],[len(pre_path),1]),marker='^', s=100)
            # ax.scatter(fut_path[:,0], fut_path[:,1], c=np.tile(c[2],[len(fut_path),1]),marker='^')
            # ax.scatter(pred_path[:,0], pred_path[:,1], c=np.tile(c[3],[len(pred_path),1]),marker='^')
            # ax.scatter(adv_pred_path[:,0], adv_pred_path[:,1], c=np.tile(c[4],[len(adv_pred_path),1]),marker='^')
            # ax.scatter(adv_agent_pre_path[:,0], adv_agent_pre_path[:,1], c=np.tile(c[1],[len(adv_agent_pre_path),1]),marker='^')

            c = ['C0','C1','C2','C3','C4']

            pre_lines.append((ax.plot(pre_path[:,0], pre_path[:,1], color=c[0],marker='^', linewidth=2)[0],pre_path))
            adv_agent_pre_line = ax.plot(adv_agent_pre_path[:,0], adv_agent_pre_path[:,1], color=c[4],marker='^')[0]
            fut_lines.append((ax.plot(fut_path[:,0], fut_path[:,1], color=c[1],marker='^', linewidth=2)[0],fut_path))
            pred_lines.append((ax.plot(pred_path[:,0], pred_path[:,1], color=c[2],marker='^', linewidth=2)[0],pred_path))
            adv_pred_lines.append((ax.plot(adv_pred_path[:,0], adv_pred_path[:,1], color=c[3],marker='^')[0],adv_pred_path))




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

                # c = np.array([[1, 1, 0, 0.8], [0, 1, 0, 0.8],[0, 0, 1, 0.8],[1, 0, 0, 0.8]])
                # lines = [pre_path,fut_path,pred_path,adv_pred_path]
                # lc = mc.LineCollection(lines, colors=c, linewidths=2)
                # ax.add_collection(lc)

                # ax.scatter(pre_path[:,0], pre_path[:,1], c=np.tile(c[0],[len(pre_path),1]))
                # ax.scatter(fut_path[:,0], fut_path[:,1], c=np.tile(c[1],[len(fut_path),1]))
                # ax.scatter(pred_path[:,0], pred_path[:,1], c=np.tile(c[2],[len(pred_path),1]))
                # ax.scatter(adv_pred_path[:,0], adv_pred_path[:,1], c=np.tile(c[3],[len(adv_pred_path),1]))
                
                pre_lines.append((ax.plot(pre_path[:,0], pre_path[:,1], color=c[0],marker='o', linewidth=2)[0],pre_path))
                fut_lines.append((ax.plot(fut_path[:,0], fut_path[:,1], color=c[1],marker='o', linewidth=2)[0],fut_path))
                pred_lines.append((ax.plot(pred_path[:,0], pred_path[:,1], color=c[2],marker='o', linewidth=2)[0],pred_path))
                adv_pred_lines.append((ax.plot(adv_pred_path[:,0], adv_pred_path[:,1], color=c[3],marker='o', linewidth=2)[0],adv_pred_path))



            legend_elements = [Line2D([0], [0], color=c[0], lw=2, label='pre'),
                            Line2D([0], [0], color=c[1], lw=2, label='gt_fut'),
                            Line2D([0], [0], color=c[2], lw=2, label='pred_fut'),
                            Line2D([0], [0], color=c[3], lw=2, label='adv_pred_fut'),
                            Line2D([0], [0], color=c[4], lw=2, label='adv_pre'),
                    Line2D([0], [0], color='w', markerfacecolor='black', marker='^', label='adv agent'),
                    Line2D([0], [0], color='w', markerfacecolor='black', marker='o', label='benign agent')]
            ax.autoscale()
            ax.margins(0.1)
            ax.legend(handles=legend_elements)
            
            def update(num, adv):
                num += 1
                if num <= cfg.past_frames:
                    for line, line_data in pre_lines:
                        line.set_data(line_data[:num,0],line_data[:num,1])
                    for line, line_data in pred_lines:
                        line.set_data([],[])
                    for line, line_data in adv_pred_lines:
                            line.set_data([],[])  
                    for line, line_data in fut_lines:
                        line.set_data([],[])
                    if adv:
                        adv_agent_pre_line.set_data(adv_agent_pre_path[:num,0],adv_agent_pre_path[:num,1])
                    else:
                        adv_agent_pre_line.set_data([],[])   
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

    print_log(f'\n\n total_num_pred: {total_num_pred}', log)
    if cfg.dataset == 'nuscenes_pred':
        scene_num = {
            'train': 32186,
            'val': 8560,
            'test': 9041
        }
        # assert total_num_pred == scene_num[generator.split]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=False)
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
    if args.epochs is None:
        epochs = [cfg.get_last_epoch()]
    else:
        epochs = [int(x) for x in args.epochs.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)

    # enable grad for pgd attack
    # torch.set_grad_enabled(False)
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

        for split in data_splits:  
            generator = data_generator(cfg, log, split=split, phase='testing')
            save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
            eval_dir = f'{save_dir}/samples_baseline'
            adv_eval_dir = f'{save_dir}/samples_adv'
            if not args.cached:
                test_model(generator, save_dir, cfg)

            log_file = os.path.join(cfg.log_dir, 'log_eval_baseline.txt')
            cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --data {split} --log {log_file}"
            subprocess.run(cmd.split(' '))
            
            log_file = os.path.join(cfg.log_dir, 'log_eval_adv.txt')
            cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {adv_eval_dir} --data {split} --log {log_file}"
            subprocess.run(cmd.split(' '))

            # remove eval folder to save disk space
            if args.cleanup:
                shutil.rmtree(save_dir)


