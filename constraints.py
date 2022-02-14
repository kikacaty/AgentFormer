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

from matplotlib import collections  as mc
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  
from matplotlib.lines import Line2D
from pdb import set_trace as st

from utils.attack_utils.constraint import DynamicModel, DynamicStats
from utils.attack_utils.attack import Attacker

import re


def get_model_prediction(data, sample_k):
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
    


def get_adv_model_prediction(data, sample_k, alpha = 1e-3, eps = 1, iterations=10):
    model.set_data(data)

    in_data = data
    if iterations == 0:
        recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)

        sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
        sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()

        return recon_motion_3D, sample_motion_3D

    adv_cfg.device = device
    adv_cfg.sample_k = sample_k
    adv_cfg.traj_scale = cfg.traj_scale

    attacker = Attacker(model, adv_cfg)
    attacker.perturb(in_data)

    adv_recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)

    adv_sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    adv_sample_motion_3D = adv_sample_motion_3D.transpose(0, 1).contiguous()

    # ori_pre_motion = model.data['pre_vel']
    # perturb_mask = torch.zeros_like(model.data['pre_vel'])
    # perturb_mask[:,0,:] = 1
    # target_id = 1
    # pre_len = model.data['pre_motion'].size()[0]

    # for i in range(iterations):
    #     perturbed_pre_motion = model.data['pre_vel'].data
    #     # for key in model.data.keys():
    #     #     if model.data[key]
    #     #     model.data[key].requires_grad = False
    #     model.data['pre_vel'].requires_grad = True
    #     target_fut_motion = model.data['fut_motion'][:,1:,:].transpose(0, 1)

    #     recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)

    #     sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    #     sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()

    #     # target_pred_motion = sample_motion_3D[0][1:]
    #     target_pred_motion = recon_motion_3D.contiguous()[1:]
    #     model.zero_grad()

    #     loss_bending = motion_bending_loss(target_fut_motion, target_pred_motion).to(device)

    #     # vel dependent motion
    #     pre_motion_from_vel = model.data['pre_motion'][-1,...].repeat(pre_len,1,1)
    #     pre_motion_from_vel[:-1,...] -= torch.cumsum(model.data['pre_vel'],dim=0) 

    #     loss_col = collision_loss(pre_motion_from_vel, adv_id=0)
    #     cost = loss_bending + loss_col * 1e-1
    #     print("Loss: {0:3f}, ({1:3f}, {2:3f})".format(cost.item(),loss_bending.item(), loss_col.item()))
    #     cost.backward()

    #     vel_mask = model.data['pre_mask'][:,:-1].t().unsqueeze(-1).tile(1,1,2) * perturb_mask
    #     adv_pre_motion = perturbed_pre_motion + alpha*vel_mask*model.data['pre_vel'].grad.sign()
    #     eta = torch.clamp(adv_pre_motion - ori_pre_motion, min=-eps, max=eps).detach_()        
    #     eta_np = eta.cpu().data
        
    #     # fix t - len(past_traj)
    #     # update_pre_motion = torch.zeros_like(model.data['pre_motion'])
    #     # update_pre_motion[1:,...] = torch.cumsum(pre_vel,dim=0) 
    #     # update_pre_motion += model.data['pre_motion'][0,...]
    #     # model.data['heading'] = torch.atan2(pre_vel[-1,0,:])

    #     # fix t
    #     update_pre_motion = model.data['pre_motion'][-1,...].repeat(pre_len,1,1)
    #     update_pre_motion[:-1,...] -= torch.cumsum(pre_vel,dim=0) 

    #     # check dynamics 
    #     pre_xy = model.data['pre_motion'][:,0,:][model.data['pre_mask'][0] == 1].cpu().data
    #     pre_xy *= cfg.traj_scale
    #     adv_h = in_data['heading'][0]
    #     adv_dynamics = DynamicModel(pre_xy, adv_h, dt_scale=2)
    #     update_pre_motion = adv_dynamics.apply_constraints(eta_np)


    #     model.data['pre_motion'] = torch.from_numpy(update_pre_motion).to(device)
    #     model.update_data(model.data, in_data)


    return adv_recon_motion_3D, adv_sample_motion_3D

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

def attack_model(generator, save_dir, cfg):
    total_num_pred = 0
    total_num_frame = 0
    dist_stats = torch.zeros(6)
    dist_cat = torch.tensor([5,6,7,8,9,10],dtype=torch.float)

    dynamic_stats = DynamicStats()

    while not generator.is_epoch_end():
        data = generator()
        if data is None or len(data['valid_id']) < 2:
            continue

        dynamic_stats.parse_dynamics(data['pre_motion_3D']* cfg.traj_scale)

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
        print('attacking ttl frame: %06d                \r' % (total_num_frame))  

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

        # skip short adv history scenarios
        if 0 in data['pre_motion_mask'][0]:
            continue

        # skip low speed scenarios
        adv_pre = data['pre_motion_3D'][0]
        adv_pre_s = adv_pre[1:] - adv_pre[:-1]
        if adv_pre_s.norm(dim=1).max() < 5/cfg.traj_scale:
            continue
        
        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)

        recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

        # generate adv
        adv_recon_motion, adv_sample_motion = get_adv_model_prediction(data, cfg.sample_k, iterations=20, alpha = 1e-2, eps=1e-2)
        adv_recon_motion_3D, adv_sample_motion_3D = adv_recon_motion * cfg.traj_scale, adv_sample_motion * cfg.traj_scale
        
        # adv_agent_pre_path = np.array([(model.data['pre_vel'][:i,0,:].sum(axis=0) + model.data['pre_motion'][0,0,:]).cpu().numpy() for i in range(len(model.data['pre_vel'])+1)])
        adv_agent_pre_path = model.data['pre_motion'][:,0,:].cpu().numpy()
        adv_agent_pre_path = adv_agent_pre_path * cfg.traj_scale

        # remove grad
        adv_recon_motion_3D = adv_recon_motion_3D.detach()
        adv_sample_motion_3D = adv_sample_motion_3D.detach()

        data['pre_motion_3D'] = model.data['pre_motion'].transpose(1,0).detach().cpu()
        data['adv_fut_motion_3D'] = adv_sample_motion[0].detach().cpu()
        data['scene_map_vis'].visualize_data(data)


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
    parser.add_argument('--adv_cfg', default='base')
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=False)
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
    adv_cfg = AdvConfig(args.adv_cfg)
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
            
            dynamic_stats = DynamicStats()

            while not generator.is_epoch_end():
                data = generator()
                if data is None or len(data['valid_id']) < 2:
                    continue

                if 0 in data['pre_motion_mask'][0]:
                    continue

                pre_motion = torch.stack(data['pre_motion_3D'], dim=0)* cfg.traj_scale
                pre_vel = (pre_motion[:,1:,:] - pre_motion[:,:-1,:]).norm(dim=-1) * 2

                if (pre_vel < 10).sum() != 0:
                    continue

                dynamic_stats.parse_dynamics(torch.stack(data['pre_motion_3D'], dim=0)* cfg.traj_scale)


                continue


