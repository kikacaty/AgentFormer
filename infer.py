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
from pdb import set_trace as st


""" Vis Utils """
from utils.tools import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

import imageio

""" map generation """
from data.map import GeometricMap, MapUtils

""" attack utils """
from attack import get_adv_model_prediction, get_model_prediction

""" Eval utils """
from eval import compute_FDE,compute_ADE,compute_MR,compute_ORR,AverageMeter
import torch.autograd.functional as AF

def pt_rbf(input, center = 0.0, scale = 1.0):
    """Assuming here that input is of shape (..., D), with center and scale of broadcastable shapes.
    """
    return torch.exp(-0.5*torch.square(input - center).sum(-1)/scale)

def prediction_reward(ego_poses, pred_motion):
    comp_dists = torch.linalg.norm(ego_poses - pred_motion, dim=-1) # Distance from predictions
    min_comps_dist = torch.amin(comp_dists, dim=1) # Minimum over time -> closest encounter
    pred_probs = 1/pred_motion.size(1)
    expected_min_dist = torch.sum(min_comps_dist * pred_probs, dim=-1) # Factoring in traj. probabilities -> expected closest encounter

    return -0.241 * pt_rbf(expected_min_dist, scale=2)

def update_eval_metrics(fut_motion, sample_motion, stats_meter=None, PI_stats_meter=None, ego_id = 0, scene_map=None, stats_func=None):
    # get sensitivity
    ego_poses = fut_motion[ego_id].to(device) * cfg.traj_scale
    pred_motions = sample_motion.transpose(0,1)[torch.arange(sample_motion.size(1))!=ego_id]

    pred_sensitivities = torch.zeros_like(pred_motions)
    
    for pred_idx in range(pred_motions.size(0)):
        pred_sensitivities[pred_idx] = AF.jacobian(lambda x: prediction_reward(ego_poses,x), pred_motions[pred_idx], create_graph=False, vectorize=True)
    
    sens_mags = torch.amax(torch.linalg.norm(pred_sensitivities, dim=-1), dim=-2)
    pred_sens = 1 + torch.nn.functional.softmax(torch.amax(sens_mags, dim=1))
    
    # evaluation metrics
    pred_motion = pred_motions.cpu() 
    fut_motion = torch.stack(fut_motion)[torch.arange(sample_motion.size(1))!=ego_id].cpu() * cfg.traj_scale
    for stats_name, meter in stats_meter.items():
        func = stats_func[stats_name]
        if stats_name == 'OffRoadRate':
            value = func(pred_motion, scene_map)
        else:
            value = func(pred_motion, fut_motion)
        meter.update(value, n=len(pred_motion))

    for stats_name, meter in PI_stats_meter.items():
        func = stats_func[stats_name]
        if stats_name == 'OffRoadRate':
            value = func(pred_motion, scene_map, PI=pred_sens)
        else:
            value = func(pred_motion, fut_motion, PI=pred_sens)
        meter.update(value, n=len(pred_motion))

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

def viz_model(args, cfg):
    # load nuscenes
    conf = read_conf('cfg/planner/rule_based.yml')
    nusc = NuScenes(version=f'v1.0-{args.version}',
                dataroot=os.path.join(args.dataroot, args.version),
                verbose=True)
    nmaps = get_nusc_maps(os.path.join(args.dataroot, args.version))
    lane_graphs = {map_name: process_lanegraph(nmap, conf['res_meters'], conf['eps'])
                            for map_name,nmap in nmaps.items()}
    scene2data = trajectory_parse(nusc, is_train=False, egoonly=False)
    keptclasses = set(conf['keptclasses'])

    # visualization settings
    window = 90

    # scenes = tqdm(scene2data)
    scenes = scene2data

    if args.adv:
        vis_dir = os.path.join(args.results_dir, 'vis_adv')
    else:
        vis_dir = os.path.join(args.results_dir, 'vis')

    os.makedirs(vis_dir, exist_ok=True )

    for k in scenes:
        # scenes.set_description(f"Processing Scene {k}")
        v = scene2data[k]
        ts = np.arange(v['objs']['ego']['traj'][0]['t'],
                       v['objs']['ego']['traj'][-1]['t'],
                       conf['dt'])
        lane_graph = lane_graphs[v['map_name']]

        scene_dir = os.path.join(vis_dir, k)

        filenames = []

        ts_bar = tqdm(ts[:-cfg.future_frames])
        ts_bar.set_description(f"Processing Scene {k}")

        for ti,t in enumerate(ts_bar):
            fig = plt.figure(figsize=(8, 8))
            gs = mpl.gridspec.GridSpec(1, 1, left=0.04, right=0.96, top=0.96, bottom=0.04)

            ax = plt.subplot(gs[0, 0])
            # plt.plot(lane_graph['edges'][:,0], lane_graph['edges'][:,1], '.', markersize=2)
            plt.plot(lane_graph['edges'][:,0], lane_graph['edges'][:,1], '.', markersize=2)
            mag = 0.3
            plt.plot(lane_graph['edges'][:,0] + mag*lane_graph['edges'][:,2],
                    lane_graph['edges'][:,1] + mag*lane_graph['edges'][:,3], '.', markersize=1)

            data = {
                'pre_motion_3D': [],
                'fut_motion_3D': [],
                'fut_motion_mask': [],
                'pre_motion_mask': [],
                # 'pre_data': pre_data,
                # 'fut_data': fut_data,
                'heading': [],
                'valid_id': [],
                'traj_scale': cfg.traj_scale,
                'pred_mask': [],
                'scene_map': None,
                'seq': k,
                'frame': ti
            }

            all_xys = []

            for objid,obj in v['objs'].items():
                if 'interp' in obj  and obj['k'] in keptclasses and obj['traj'][0]['t'] <= t <= obj['traj'][-1]['t']:
                    x,y,hcos,hsin = obj['interp'](t)
                    h = np.arctan2(hsin, hcos)
                    matches = get_lane_matches(x, y, h, lane_graph,
                                            cdistmax=1.0 - np.cos(np.radians(conf['cdisttheta'])),
                                            xydistmax=conf['xydistmax'])
                    final_matches = cluster_matches_combine(x, y, matches, lane_graph)

                    backdist, fordist = 4.0, 16.0 

                    # prediction splines
                    all_splines = get_prediction_splines(final_matches, lane_graph, backdist=backdist, fordist=fordist,
                                                        xydistmax=conf['xydistmax'], egoxy=np.array([x,y]),
                                                        lane_ds=conf['lane_ds'], lane_sig=conf['lane_sig'], sbuffer=conf['sbuffer'],
                                                        egoh=h)
                    if objid == 'ego':
                        for spline in all_splines:
                            xyhs = spline(np.arange(-backdist, fordist))
                            for lx,ly,lhcos,lhsin in xyhs:
                                plot_car(lx, ly, np.arctan2(lhsin, lhcos), obj['l']*0.4, obj['w']*0.4, color='yellow', alpha=0.3)

                    else:
                        plt.plot(final_matches['closest'][:,0], final_matches['closest'][:,1], '.')
                        carcolor = 'tab:blue' if final_matches['closest'].shape[0] > 0 else 'tab:orange'
                        plot_car(x, y, h, obj['l'], obj['w'], color=carcolor)
                        if final_matches['closest'].shape[0] == 0: 
                            continue

                    # gather prediction data
                    if ti+1 >= cfg.past_frames:
                        xys = []
                        xy_masks = []
                        for past_frame in range(cfg.past_frames):
                            past_t = ts[ti-past_frame]
                            if 'interp' in obj  and obj['k'] in keptclasses and obj['traj'][0]['t'] <= past_t <= obj['traj'][-1]['t']:
                                x,y,hcos,hsin = obj['interp'](ts[ti - past_frame])
                                xy_masks.append(1.)
                            else:
                                # use the same x as history
                                xy_masks.append(0.)
                            xys.append([x,y])
                        
                        all_xys += xys
                        xys.reverse()
                        xy_masks.reverse()

                        xys = np.array(xys,dtype=np.float32)
                        xys = torch.from_numpy(xys)
                        xy_masks = np.array(xy_masks, dtype=np.float32)
                        xy_masks = torch.from_numpy(xy_masks)

                        data['pre_motion_3D'].append(xys/cfg.traj_scale)
                        data['pre_motion_mask'].append(xy_masks)

                        xys = []
                        xy_masks = []
                        for fut_frame in range(cfg.future_frames):
                            fut_t = ts[ti+fut_frame]
                            if 'interp' in obj  and obj['k'] in keptclasses and obj['traj'][0]['t'] <= fut_t <= obj['traj'][-1]['t']:
                                x,y,hcos,hsin = obj['interp'](ts[ti + fut_frame])
                                xy_masks.append(1.)
                            else:
                                # use the same x as history
                                xy_masks.append(0.)
                            xys.append([x,y])

                        all_xys += xys

                        xys = np.array(xys,dtype=np.float32)
                        xys = torch.from_numpy(xys)
                        xy_masks = np.array(xy_masks.reverse(), dtype=np.float32)
                        xy_masks = torch.from_numpy(xy_masks)

                        data['fut_motion_3D'].append(xys/cfg.traj_scale)
                        data['fut_motion_mask'].append(xy_masks)
                        data['heading'].append(h)
                        data['valid_id'].append(objid)
                        data['pred_mask'].append(1.0)
            
            
            # plot ego
            ego_obj = v['objs']['ego']
            x,y,hcos,hsin = ego_obj['interp'](t)
            h = np.arctan2(hsin, hcos)
            plot_car(x, y, h, obj['l'], obj['w'], color='tab:green')
            
            
            # making predictions if enough past frames
            if len(all_xys) > 0:
                # generating maps
                nusc_map = nmaps[v['map_name']]
                scale = 3.0
                margin = 75
                xy = np.array(all_xys).astype(np.float32)
                x_min = np.round(xy[:, 0].min() - margin)
                x_max = np.round(xy[:, 0].max() + margin)
                y_min = np.round(xy[:, 1].min() - margin)
                y_max = np.round(xy[:, 1].max() + margin)
                x_size = x_max - x_min
                y_size = y_max - y_min
                patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)
                patch_angle = 0
                canvas_size = (np.round(scale * y_size).astype(int), np.round(scale * x_size).astype(int))
                homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
                layer_names = MapUtils.layer_names
                colors = MapUtils.colors

                map_mask = (nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0).astype(np.uint8)
                map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
                map_mask_vehicle = np.stack((np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=0)

                # scene_map = np.transpose(map_mask_vehicle, (0,2,1))
                scene_map = map_mask_vehicle

                meta = np.array([x_min, y_min, scale])
                map_origin = meta[:2]
                scale = meta[2]
                homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
                geom_scene_map = GeometricMap(scene_map, homography, map_origin)

                data['scene_map'] = geom_scene_map

                # if data is None:
                #     continue
                # seq_name, frame = data['seq'], data['frame']
                # frame = int(frame)
                # sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
                # sys.stdout.flush()

                gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
                if args.adv:
                    recon_motion_3D, sample_motion_3D = get_adv_model_prediction(data, cfg.sample_k, model, adv_cfg=adv_cfg)
                else:
                    with torch.no_grad():
                        recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k, model)
                recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

                # remove grad
                recon_motion_3D = recon_motion_3D.detach()
                sample_motion_3D = sample_motion_3D.detach()

                # plot predictions
                all_pred_trajs = sample_motion_3D.cpu().numpy()[0]
                all_pred_trajs = np.stack(all_pred_trajs)
                for idx in range(all_pred_trajs.shape[0]):
                    traj = all_pred_trajs[idx]
                    pre_traj = data['pre_motion_3D'][idx].cpu().numpy() * cfg.traj_scale
                    full_traj = np.concatenate([pre_traj, traj])[3:]
                    xyhs = xy2xyhs(full_traj,0,data['heading'][idx])
                    obj = v['objs'][data['valid_id'][idx]]
                    for lx,ly,lhcos,lhsin in xyhs:
                        plot_car(lx, ly, np.arctan2(lhsin, lhcos), obj['l']*0.4, obj['w']*0.4, color='tab:blue', alpha=0.3)

            # ego
            ego_obj = v['objs']['ego']
            x,y,hcos,hsin = ego_obj['interp'](t)
            h = np.arctan2(hsin, hcos)
            plot_car(x, y, h, ego_obj['l'], ego_obj['w'], color='tab:green')
            centerx,centery = x, y
            plt.xlim((centerx - window, centerx + window))
            plt.ylim((centery - window, centery + window))
            plt.grid(b=None)
            ax.set_aspect('equal')
            imname = f'{vis_dir}/{k}_{ti:04}.jpg'

            # map for visualization
            patch_angle = 0
            patch_box_viz = (centerx, centery, window * 2, window * 2)
            res_scale = 5
            canvas_size = (window*2*res_scale, window*2*res_scale)
            nusc_map = nmaps[v['map_name']]
            map_mask_viz = (nusc_map.get_map_mask(patch_box_viz, patch_angle, MapUtils.layer_names, canvas_size) * 255.0).astype(np.uint8)
            map_mask_viz = np.swapaxes(map_mask_viz, 1, 2)  # x axis comes first

            map_mask_plot = np.ones_like(map_mask_viz[:3])
            map_mask_plot[:] = np.array(MapUtils.colors['rest'])[:, None, None]
            for layer in ['lane', 'road_segment', 'drivable_area', 'road_divider', 'ped_crossing', 'walkway']:
                xind, yind = np.where(map_mask_viz[MapUtils.layer_names.index(layer)])
                map_mask_plot[:, xind, yind] = np.array(MapUtils.colors[layer])[:, None]
            map_mask_plot = np.transpose(map_mask_plot, (2,1,0))
            map_mask_plot = np.flip(map_mask_plot,0)

            plt.imshow(map_mask_plot, alpha=0.8, extent=[plt.xlim()[0], plt.xlim()[1], plt.ylim()[0], plt.ylim()[1]])

            plt.savefig(imname)
            plt.close(fig)

            filenames.append(imname)

        # build gif
        if len(filenames) > 0:
            gifname = f'{vis_dir}/{k}.gif'
            with imageio.get_writer(gifname, mode='I') as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            print('saving gif', gifname)

            # Remove files
            for filename in set(filenames):
                os.remove(filename)

def drive_model(args, cfg):
    # load nuscenes
    conf = read_conf('cfg/planner/rule_based.yml')
    nusc = NuScenes(version=f'v1.0-{args.version}',
                dataroot=os.path.join(args.dataroot, args.version),
                verbose=True)
    nmaps = get_nusc_maps(os.path.join(args.dataroot, args.version))
    lane_graphs = {map_name: process_lanegraph(nmap, conf['res_meters'], conf['eps'])
                            for map_name,nmap in nmaps.items()}
    scene2data = trajectory_parse(nusc, is_train=False, egoonly=False)
    keptclasses = set(conf['keptclasses'])

    # visualization settings
    window = 90

    # scenes = tqdm(scene2data)
    scenes = scene2data

    if args.adv:
        vis_dir = os.path.join(args.results_dir, f'dive_adv/{args.version}')
    else:
        vis_dir = os.path.join(args.results_dir, f'drive/{args.version}')

    os.makedirs(vis_dir, exist_ok=True)

    if args.version == 'trainval':
        scenes = ['scene-0003', 'scene-0099', 'scene-0101', 'scene-0102', 'scene-0103', 'scene-0108',
            'scene-0271', 'scene-0329', 'scene-0331', 'scene-0520', 'scene-0524', 'scene-0556', 'scene-0559',
            'scene-0560', 'scene-0907', 'scene-0912', 'scene-0922', 'scene-0966', 'scene-0635']

    for k in scenes:
        # scenes.set_description(f"Processing Scene {k}")
        v = scene2data[k]
        ts = np.arange(v['objs']['ego']['traj'][0]['t'],
                       v['objs']['ego']['traj'][-1]['t'],
                       conf['dt'])
        lane_graph = lane_graphs[v['map_name']]

        scene_dir = os.path.join(vis_dir, k)

        filenames = []

        ts_bar = tqdm(ts[:-cfg.future_frames])
        ts_bar.set_description(f"Processing Scene {k}")

        controlids = ['ego']
        wstate = None

        cache_wstates = []

        for ti,t in enumerate(ts_bar):
            fig = plt.figure(figsize=(8, 8))
            gs = mpl.gridspec.GridSpec(1, 1, left=0.04, right=0.96, top=0.96, bottom=0.04)

            ax = plt.subplot(gs[0, 0])
            # plot laneline
            plt.plot(lane_graph['edges'][:,0], lane_graph['edges'][:,1], '.', markersize=2)
            mag = 0.3
            plt.plot(lane_graph['edges'][:,0] + mag*lane_graph['edges'][:,2],
                    lane_graph['edges'][:,1] + mag*lane_graph['edges'][:,3], '.', markersize=1)

            data = {
                'pre_motion_3D': [],
                'fut_motion_3D': [],
                'fut_motion_mask': [],
                'pre_motion_mask': [],
                # 'pre_data': pre_data,
                # 'fut_data': fut_data,
                'heading': [],
                'valid_id': [],
                'traj_scale': cfg.traj_scale,
                'pred_mask': [],
                'scene_map': None,
                'seq': k,
                'frame': ti
            }

            all_xys = []

            for objid,obj in v['objs'].items():
                if 'interp' in obj  and obj['k'] in keptclasses and obj['traj'][0]['t'] <= t and t + conf['dt'] <= obj['traj'][-1]['t']:
                    x,y,hcos,hsin = obj['interp'](t)
                    h = np.arctan2(hsin, hcos)
                    matches = get_lane_matches(x, y, h, lane_graph,
                                            cdistmax=1.0 - np.cos(np.radians(conf['cdisttheta'])),
                                            xydistmax=conf['xydistmax'])
                    final_matches = cluster_matches_combine(x, y, matches, lane_graph)

                    # backdist, fordist = 1.0, 16.0 

                    # prediction splines
                    # all_splines = get_prediction_splines(final_matches, lane_graph, backdist=backdist, fordist=fordist,
                    #                                     xydistmax=conf['xydistmax'], egoxy=np.array([x,y]),
                    #                                     lane_ds=conf['lane_ds'], lane_sig=conf['lane_sig'], sbuffer=conf['sbuffer'],
                    #                                     egoh=h)
                    if objid == 'ego':
                        plot_car(x, y, h, obj['l'], obj['w'], color='green')
                        # if wstate is None:
                        #     for spline in all_splines:
                        #         xyhs = spline(np.arange(-backdist, fordist))
                        #         for lx,ly,lhcos,lhsin in xyhs:
                        #             plot_car(lx, ly, np.arctan2(lhsin, lhcos), obj['l']*0.4, obj['w']*0.4, color='yellow', alpha=0.3)

                    else:
                        plt.plot(final_matches['closest'][:,0], final_matches['closest'][:,1], '.')
                        carcolor = 'tab:blue' if final_matches['closest'].shape[0] > 0 else 'tab:orange'
                        plot_car(x, y, h, obj['l'], obj['w'], color=carcolor)
                        if final_matches['closest'].shape[0] == 0: 
                            continue

                    # gather prediction data
                    if ti+1 >= cfg.past_frames:
                        xys = []
                        xy_masks = []
                        for past_frame in range(cfg.past_frames):
                            past_t = ts[ti-past_frame]
                            if 'interp' in obj  and obj['k'] in keptclasses and obj['traj'][0]['t'] <= past_t <= obj['traj'][-1]['t']:
                                x,y,hcos,hsin = obj['interp'](ts[ti - past_frame])
                                xy_masks.append(1.)
                            else:
                                # use the same x as history
                                xy_masks.append(0.)
                            xys.append([x,y])
                        
                        all_xys += xys
                        xys.reverse()
                        xy_masks.reverse()

                        xys = np.array(xys,dtype=np.float32)
                        xys = torch.from_numpy(xys)
                        xy_masks = np.array(xy_masks, dtype=np.float32)
                        xy_masks = torch.from_numpy(xy_masks)

                        data['pre_motion_3D'].append(xys/cfg.traj_scale)
                        data['pre_motion_mask'].append(xy_masks)

                        xys = []
                        xy_masks = []
                        for fut_frame in range(cfg.future_frames):
                            fut_t = ts[ti+fut_frame]
                            if 'interp' in obj  and obj['k'] in keptclasses and obj['traj'][0]['t'] <= fut_t <= obj['traj'][-1]['t']:
                                x,y,hcos,hsin = obj['interp'](ts[ti + fut_frame])
                                xy_masks.append(1.)
                            else:
                                # use the same x as history
                                xy_masks.append(0.)
                            xys.append([x,y])

                        all_xys += xys

                        xys = np.array(xys,dtype=np.float32)
                        xys = torch.from_numpy(xys)
                        xy_masks = np.array(xy_masks.reverse(), dtype=np.float32)
                        xy_masks = torch.from_numpy(xy_masks)

                        data['fut_motion_3D'].append(xys/cfg.traj_scale)
                        data['fut_motion_mask'].append(xy_masks)
                        data['heading'].append(h)
                        data['valid_id'].append(objid)
                        data['pred_mask'].append(1.0)
            
            
            # making predictions if enough past frames
            if len(all_xys) > 0:
                # generating maps
                nusc_map = nmaps[v['map_name']]
                scale = 3.0
                margin = 75
                xy = np.array(all_xys).astype(np.float32)
                x_min = np.round(xy[:, 0].min() - margin)
                x_max = np.round(xy[:, 0].max() + margin)
                y_min = np.round(xy[:, 1].min() - margin)
                y_max = np.round(xy[:, 1].max() + margin)
                x_size = x_max - x_min
                y_size = y_max - y_min
                patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)
                patch_angle = 0
                canvas_size = (np.round(scale * y_size).astype(int), np.round(scale * x_size).astype(int))
                homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
                layer_names = MapUtils.layer_names
                colors = MapUtils.colors

                map_mask = (nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0).astype(np.uint8)
                map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
                map_mask_vehicle = np.stack((np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=0)

                # scene_map = np.transpose(map_mask_vehicle, (0,2,1))
                scene_map = map_mask_vehicle

                meta = np.array([x_min, y_min, scale])
                map_origin = meta[:2]
                scale = meta[2]
                homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
                geom_scene_map = GeometricMap(scene_map, homography, map_origin)

                data['scene_map'] = geom_scene_map

                gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
                if args.adv:
                    adv_recon_motion_list, adv_sample_motion_list = get_adv_model_prediction(data, cfg.sample_k, model, adv_cfg=cfg.adv_cfg)
                    recon_motion_3D, sample_motion_3D = adv_recon_motion_list[-1], adv_sample_motion_list[-1]
                else:
                    with torch.no_grad():
                        recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k, model)
                recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

                # remove grad
                recon_motion_3D = recon_motion_3D.detach()
                sample_motion_3D = sample_motion_3D.detach()

                # plot predictions
                all_pred_trajs_samples = sample_motion_3D.cpu().numpy()
                if conf['vis_pred']:
                    for sample in range(all_pred_trajs_samples.shape[0]):
                        all_pred_trajs = all_pred_trajs_samples[sample]
                        for idx in range(all_pred_trajs.shape[0]):
                            if data['valid_id'][idx] == 'ego':
                                continue
                            traj = all_pred_trajs[idx]
                            pre_traj = data['pre_motion_3D'][idx].cpu().numpy() * cfg.traj_scale
                            full_traj = np.concatenate([pre_traj, traj])[3:]
                            xyhs = xy2xyhs(full_traj,0,data['heading'][idx])
                            obj = v['objs'][data['valid_id'][idx]]
                            for lx,ly,lhcos,lhsin in xyhs:
                                plot_car(lx, ly, np.arctan2(lhsin, lhcos), obj['l']*0.4, obj['w']*0.4, color='tab:blue', alpha=0.1/all_pred_trajs_samples.shape[0])

                # ================= Drive start =================
                if wstate is None:
                    # init wstate when started prediction
                    t0 = t
                    t1 = t0 + conf['dt']
                    wstate = get_init_wstate(t0, t1, v, keptclasses)

                    # # check predictions
                    # for objid in wstate['objs'].keys():
                    #     if objid != 'ego' and objid not in data['valid_id']:
                    #         print(f"Missing predictions of: {objid}")
                    #         st()

                compute_splines(wstate, lane_graph, conf['cdisttheta'], conf['xydistmax'],
                                conf['lane_ds'], conf['lane_sig'], conf['sbuffer'], conf['smax'], conf['nsteps']*conf['preddt'])
                
                plan_other_trajs = [] # collect othertrajs for planner
                for sample in range(all_pred_trajs_samples.shape[0]):
                    all_pred_trajs = all_pred_trajs_samples[sample]
                    for idx in range(all_pred_trajs.shape[0]):
                        if data['valid_id'][idx] == 'ego' or data['valid_id'][idx] not in wstate['objs'].keys():
                            continue
                        traj = all_pred_trajs[idx]
                        pre_traj = data['pre_motion_3D'][idx].cpu().numpy() * cfg.traj_scale
                        full_traj = np.concatenate([pre_traj, traj])[3:]
                        xyhs = xy2xyhs(full_traj,0,data['heading'][idx])
                        obj = v['objs'][data['valid_id'][idx]]
                        traj = np.empty((conf['nsteps']+1, 5))
                        traj[:, :2] = xyhs[:, :2]
                        traj[:, 2] = np.arctan2(xyhs[:, 3], xyhs[:, 2])
                        traj[:, 3] = obj['l']
                        traj[:, 4] = obj['w']
                        plan_other_trajs.append(traj)

                for controlid in controlids:
                    compute_action_with_prediction(wstate, controlid, conf['dt'], conf['nsteps'], conf['preddt'], conf['predsfacs'], conf['predafacs'],
                                conf['interacdist'], conf['maxacc'], f'plan{k}_{0:04}', lane_graph, conf['planaccfacs'], conf['smax'],
                                conf['plannspeeds'], debug=False, score_wmin=conf['score_wmin'], score_wfac=conf['score_wfac'], other_objs=plan_other_trajs)

                plan_ego_obj = wstate['objs']['ego']
                # for interp in plan_ego_obj['splines']:
                #     lane = interp(np.arange(-1, 20))
                #     for lx,ly,lhcos,lhsin in lane:
                #         plot_car(lx, ly, np.arctan2(lhsin, lhcos), plan_ego_obj['l']*0.3, plan_ego_obj['w']*0.3,
                #                 color='yellow', alpha=0.3)
                plan_ego_spline = plan_ego_obj['splines'][0]
                sprof = plan_ego_obj['control']['sprof']
                egolocs = plan_ego_spline(sprof['teval'])
                for lx,ly,lhcos,lhsin in egolocs:
                    plot_car(lx, ly, np.arctan2(lhsin, lhcos), plan_ego_obj['l']*0.3, plan_ego_obj['w']*0.3,
                            color='yellow', alpha=0.5)

                x,y,h = plan_ego_obj['x'],plan_ego_obj['y'],plan_ego_obj['h']

                plot_car(x, y, h, plan_ego_obj['l'], plan_ego_obj['w'], color='red')
                # for spline in plan_ego_obj['splines']:
                #     xyhs = spline(np.arange(-backdist, fordist))
                #     for lx,ly,lhcos,lhsin in xyhs:
                #         plot_car(lx, ly, np.arctan2(lhsin, lhcos), plan_ego_obj['l']*0.4, plan_ego_obj['w']*0.4, color='yellow', alpha=0.3)

                if conf['dump_scene']:
                    cache_wstates.append(dump_wstate(wstate, egolocs))

                wstate = update_wstate_pred(wstate, v, conf['dt'], keptclasses)

            # ================= Drive end =================

            # centering ego
            if wstate is None:
                ego_obj = v['objs']['ego']
                x,y,hcos,hsin = ego_obj['interp'](t)
                h = np.arctan2(hsin, hcos)
                
            centerx,centery = x, y
            plt.xlim((centerx - window, centerx + window))
            plt.ylim((centery - window, centery + window))
            plt.grid(b=None)
            ax.set_aspect('equal')
            imname = f'{vis_dir}/{k}_{ti:04}.png'

            # map for visualization
            patch_angle = 0
            patch_box_viz = (centerx, centery, window * 2, window * 2)
            res_scale = 5
            canvas_size = (window*2*res_scale, window*2*res_scale)
            nusc_map = nmaps[v['map_name']]
            map_mask_viz = (nusc_map.get_map_mask(patch_box_viz, patch_angle, MapUtils.layer_names, canvas_size) * 255.0).astype(np.uint8)
            map_mask_viz = np.swapaxes(map_mask_viz, 1, 2)  # x axis comes first

            map_mask_plot = np.ones_like(map_mask_viz[:3])
            map_mask_plot[:] = np.array(MapUtils.colors['rest'])[:, None, None]
            for layer in ['lane', 'road_segment', 'drivable_area', 'road_divider', 'ped_crossing', 'walkway']:
                xind, yind = np.where(map_mask_viz[MapUtils.layer_names.index(layer)])
                map_mask_plot[:, xind, yind] = np.array(MapUtils.colors[layer])[:, None]
            map_mask_plot = np.transpose(map_mask_plot, (2,1,0))
            map_mask_plot = np.flip(map_mask_plot,0)

            plt.imshow(map_mask_plot, alpha=0.8, extent=[plt.xlim()[0], plt.xlim()[1], plt.ylim()[0], plt.ylim()[1]])

            plt.savefig(imname)
            plt.close(fig)

            filenames.append(imname)

        # build gif
        if len(filenames) > 0:
            gifname = f'{vis_dir}/{k}.gif'
            with imageio.get_writer(gifname, mode='I') as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            print('saving gif', gifname)

            # Remove files
            for filename in set(filenames):
                os.remove(filename)

            if conf['dump_scene']:
                outname = f'{vis_dir}/{k}.pkl'
                with open(outname, 'wb') as writer:
                    pickle.dump(cache_wstates, writer)

def eval_model(args, cfg):
    # load nuscenes
    conf = read_conf('cfg/planner/rule_based.yml')
    nusc = NuScenes(version=f'v1.0-{args.version}',
                dataroot=os.path.join(args.dataroot, args.version),
                verbose=True)
    nmaps = get_nusc_maps(os.path.join(args.dataroot, args.version))
    lane_graphs = {map_name: process_lanegraph(nmap, conf['res_meters'], conf['eps'])
                            for map_name,nmap in nmaps.items()}
    scene2data = trajectory_parse(nusc, is_train=False, egoonly=False)
    keptclasses = set(conf['keptclasses'])

    # visualization settings
    window = 90
    vis = not args.no_vis

    # eval settings
    stats_func = {
        'ADE': compute_ADE,
        'FDE': compute_FDE,
        'MissRate': compute_MR,
        'OffRoadRate': compute_ORR,
    }

    stats_meter = {x: AverageMeter() for x in stats_func.keys()}
    PI_stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    if args.adv:
        adv_stats_meter = {x: AverageMeter() for x in stats_func.keys()}
        adv_PI_stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    # scenes = tqdm(scene2data)
    scenes = scene2data

    if args.adv:
        vis_dir = os.path.join(args.results_dir, 'eval_adv')
    else:
        vis_dir = os.path.join(args.results_dir, 'eval')

    os.makedirs(vis_dir, exist_ok=True)

    # if args.version == 'trainval':
    #     scenes = ['scene-0023', 'scene-0051', 'scene-0070', 'scene-0150', 'scene-0172', 'scene-0194', 'scene-0235', 'scene-0238', 'scene-0246', 'scene-0256', 'scene-0297', 'scene-0303', 'scene-0317', 'scene-0318', 'scene-0388', 'scene-0395', 'scene-0426', 'scene-0459', 'scene-0474', 'scene-0504', 'scene-0518', 'scene-0538', 'scene-0650', 'scene-0652', 'scene-0688', 'scene-0731', 'scene-0733', 'scene-0734', 'scene-0752', 'scene-0764', 'scene-0765', 'scene-0768', 'scene-0855', 'scene-0887', 'scene-0890', 'scene-0899', 'scene-0956', 'scene-0959', 'scene-0961', 'scene-0981', 'scene-1023', 'scene-1044', 'scene-1046', 'scene-1047', 'scene-1048', 'scene-1093', 'scene-1108']


    for k in scenes:
        # scenes.set_description(f"Processing Scene {k}")
        v = scene2data[k]
        ts = np.arange(v['objs']['ego']['traj'][0]['t'],
                       v['objs']['ego']['traj'][-1]['t'],
                       conf['dt'])
        lane_graph = lane_graphs[v['map_name']]

        scene_dir = os.path.join(vis_dir, k)

        filenames = []

        ts_bar = tqdm(ts[:-cfg.future_frames])
        ts_bar.set_description(f"Processing Scene {k}")

        for ti,t in enumerate(ts_bar):

            if vis:
                fig = plt.figure(figsize=(8, 8))
                gs = mpl.gridspec.GridSpec(1, 1, left=0.04, right=0.96, top=0.96, bottom=0.04)

                ax = plt.subplot(gs[0, 0])
                # plt.plot(lane_graph['edges'][:,0], lane_graph['edges'][:,1], '.', markersize=2)
                plt.plot(lane_graph['edges'][:,0], lane_graph['edges'][:,1], '.', markersize=2)
                mag = 0.3
                plt.plot(lane_graph['edges'][:,0] + mag*lane_graph['edges'][:,2],
                        lane_graph['edges'][:,1] + mag*lane_graph['edges'][:,3], '.', markersize=1)

            data = {
                'pre_motion_3D': [],
                'fut_motion_3D': [],
                'fut_motion_mask': [],
                'pre_motion_mask': [],
                # 'pre_data': pre_data,
                # 'fut_data': fut_data,
                'heading': [],
                'valid_id': [],
                'traj_scale': cfg.traj_scale,
                'pred_mask': [],
                'scene_map': None,
                'seq': k,
                'frame': ti
            }

            all_xys = []

            for objid,obj in v['objs'].items():
                if 'interp' in obj  and obj['k'] in keptclasses and obj['traj'][0]['t'] <= t <= obj['traj'][-1]['t']:
                    x,y,hcos,hsin = obj['interp'](t)
                    h = np.arctan2(hsin, hcos)
                    matches = get_lane_matches(x, y, h, lane_graph,
                                            cdistmax=1.0 - np.cos(np.radians(conf['cdisttheta'])),
                                            xydistmax=conf['xydistmax'])
                    final_matches = cluster_matches_combine(x, y, matches, lane_graph)

                    if objid != 'ego':
                        if vis:
                            plt.plot(final_matches['closest'][:,0], final_matches['closest'][:,1], '.')
                            carcolor = 'tab:blue' if final_matches['closest'].shape[0] > 0 else 'tab:orange'
                            plot_car(x, y, h, obj['l'], obj['w'], color=carcolor)
                        if final_matches['closest'].shape[0] == 0: 
                            continue

                    # gather prediction data
                    if ti+1 >= cfg.past_frames:
                        xys = []
                        xy_masks = []
                        for past_frame in range(cfg.past_frames):
                            past_t = ts[ti-past_frame]
                            if 'interp' in obj  and obj['k'] in keptclasses and obj['traj'][0]['t'] <= past_t <= obj['traj'][-1]['t']:
                                x,y,hcos,hsin = obj['interp'](ts[ti - past_frame])
                                xy_masks.append(1.)
                            else:
                                # use the same x as history
                                xy_masks.append(0.)
                            xys.append([x,y])
                        
                        all_xys += xys
                        xys.reverse()
                        xy_masks.reverse()

                        xys = np.array(xys,dtype=np.float32)
                        xys = torch.from_numpy(xys)
                        xy_masks = np.array(xy_masks, dtype=np.float32)
                        xy_masks = torch.from_numpy(xy_masks)

                        data['pre_motion_3D'].append(xys/cfg.traj_scale)
                        data['pre_motion_mask'].append(xy_masks)

                        xys = []
                        xy_masks = []
                        for fut_frame in range(cfg.future_frames):
                            fut_t = ts[ti+fut_frame]
                            if 'interp' in obj  and obj['k'] in keptclasses and obj['traj'][0]['t'] <= fut_t <= obj['traj'][-1]['t']:
                                x,y,hcos,hsin = obj['interp'](ts[ti + fut_frame])
                                xy_masks.append(1.)
                            else:
                                # use the same x as history
                                xy_masks.append(0.)
                            xys.append([x,y])

                        all_xys += xys

                        xys = np.array(xys,dtype=np.float32)
                        xys = torch.from_numpy(xys)
                        xy_masks = np.array(xy_masks.reverse(), dtype=np.float32)
                        xy_masks = torch.from_numpy(xy_masks)

                        data['fut_motion_3D'].append(xys/cfg.traj_scale)
                        data['fut_motion_mask'].append(xy_masks)
                        data['heading'].append(h)
                        data['valid_id'].append(objid)
                        data['pred_mask'].append(1.0)
            
            
            # making predictions if enough past frames and enough agents
            if len(data['valid_id']) > 1:
                # generating maps
                nusc_map = nmaps[v['map_name']]
                scale = 3.0
                margin = 75
                xy = np.array(all_xys).astype(np.float32)
                x_min = np.round(xy[:, 0].min() - margin)
                x_max = np.round(xy[:, 0].max() + margin)
                y_min = np.round(xy[:, 1].min() - margin)
                y_max = np.round(xy[:, 1].max() + margin)
                x_size = x_max - x_min
                y_size = y_max - y_min
                patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)
                patch_angle = 0
                canvas_size = (np.round(scale * y_size).astype(int), np.round(scale * x_size).astype(int))
                homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
                layer_names = MapUtils.layer_names
                colors = MapUtils.colors

                map_mask = (nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0).astype(np.uint8)
                map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
                map_mask_vehicle = np.stack((np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=0)

                # scene_map = np.transpose(map_mask_vehicle, (0,2,1))
                scene_map = map_mask_vehicle

                meta = np.array([x_min, y_min, scale])
                map_origin = meta[:2]
                scale = meta[2]
                homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
                geom_scene_map = GeometricMap(scene_map, homography, map_origin)

                data['scene_map'] = geom_scene_map

                gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
                if args.adv:
                    adv_recon_motion_list, adv_sample_motion_list = get_adv_model_prediction(data, cfg.sample_k, model, adv_cfg=cfg.adv_cfg)
                    adv_recon_motion_3D, adv_sample_motion_3D = adv_recon_motion_list[-1], adv_sample_motion_list[-1]
                    adv_recon_motion_3D, adv_sample_motion_3D = adv_recon_motion_3D * cfg.traj_scale, adv_sample_motion_3D * cfg.traj_scale

                    adv_recon_motion_3D = adv_recon_motion_3D.detach()
                    adv_sample_motion_3D = adv_sample_motion_3D.detach()
                
                with torch.no_grad():
                    recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k, model)
                recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

                # remove grad
                recon_motion_3D = recon_motion_3D.detach()
                sample_motion_3D = sample_motion_3D.detach()

                # plot predictions
                if vis:
                    all_pred_trajs_samples = sample_motion_3D.cpu().numpy()
                    if conf['vis_pred']:
                        all_pred_trajs = all_pred_trajs_samples[0]
                        for idx in range(all_pred_trajs.shape[0]):
                            if data['valid_id'][idx] == 'ego':
                                continue
                            traj = all_pred_trajs[idx]
                            pre_traj = data['pre_motion_3D'][idx].cpu().numpy() * cfg.traj_scale
                            full_traj = np.concatenate([pre_traj, traj])[3:]
                            xyhs = xy2xyhs(full_traj,0,data['heading'][idx])
                            obj = v['objs'][data['valid_id'][idx]]
                            for lx,ly,lhcos,lhsin in xyhs:
                                plot_car(lx, ly, np.arctan2(lhsin, lhcos), obj['l']*0.4, obj['w']*0.4, color='tab:blue', alpha=0.2/all_pred_trajs_samples.shape[0])
                    
                    all_pred_trajs_samples = adv_sample_motion_3D.cpu().numpy()
                    if conf['vis_pred']:
                        all_pred_trajs = all_pred_trajs_samples[0]
                        for idx in range(all_pred_trajs.shape[0]):
                            if data['valid_id'][idx] == 'ego':
                                continue
                            traj = all_pred_trajs[idx]
                            pre_traj = data['pre_motion_3D'][idx].cpu().numpy() * cfg.traj_scale
                            full_traj = np.concatenate([pre_traj, traj])[3:]
                            xyhs = xy2xyhs(full_traj,0,data['heading'][idx])
                            obj = v['objs'][data['valid_id'][idx]]
                            for lx,ly,lhcos,lhsin in xyhs:
                                plot_car(lx, ly, np.arctan2(lhsin, lhcos), obj['l']*0.4, obj['w']*0.4, color='tab:red', alpha=0.2/all_pred_trajs_samples.shape[0])

                # benign prediction metrics
                for idx, obs_id in enumerate(data['valid_id']):
                    if obs_id == 'ego':
                        ego_id = idx

                update_eval_metrics(data['fut_motion_3D'], sample_motion_3D, 
                                    ego_id=ego_id, scene_map=geom_scene_map, stats_func=stats_func,
                                    stats_meter = stats_meter, PI_stats_meter = PI_stats_meter)
                print('='*10 + ' Benign ' + '='*10)
                print('Base-metrics->' + ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in stats_meter.items()]))
                print('PI-metrics->' + ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in PI_stats_meter.items()]))

                if args.adv:
                    update_eval_metrics(data['fut_motion_3D'], adv_sample_motion_3D, 
                                    ego_id=ego_id, scene_map=geom_scene_map, stats_func=stats_func,
                                    stats_meter = adv_stats_meter, PI_stats_meter = adv_PI_stats_meter)
                    print('='*10 + ' Adv ' + '='*10)
                    print('Base-metrics->' + ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in adv_stats_meter.items()]))
                    print('PI-metrics->' + ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in adv_PI_stats_meter.items()]))


            # ego
            ego_obj = v['objs']['ego']
            x,y,hcos,hsin = ego_obj['interp'](t)
            h = np.arctan2(hsin, hcos)
            if vis:
                plot_car(x, y, h, ego_obj['l'], ego_obj['w'], color='tab:green')
                centerx,centery = x, y
                plt.xlim((centerx - window, centerx + window))
                plt.ylim((centery - window, centery + window))
                plt.grid(b=None)
                ax.set_aspect('equal')
                imname = f'{vis_dir}/{k}_{ti:04}.png'

                # map for visualization
                patch_angle = 0
                patch_box_viz = (centerx, centery, window * 2, window * 2)
                res_scale = 5
                canvas_size = (window*2*res_scale, window*2*res_scale)
                nusc_map = nmaps[v['map_name']]
                map_mask_viz = (nusc_map.get_map_mask(patch_box_viz, patch_angle, MapUtils.layer_names, canvas_size) * 255.0).astype(np.uint8)
                map_mask_viz = np.swapaxes(map_mask_viz, 1, 2)  # x axis comes first

                map_mask_plot = np.ones_like(map_mask_viz[:3])
                map_mask_plot[:] = np.array(MapUtils.colors['rest'])[:, None, None]
                for layer in ['lane', 'road_segment', 'drivable_area', 'road_divider', 'ped_crossing', 'walkway']:
                    xind, yind = np.where(map_mask_viz[MapUtils.layer_names.index(layer)])
                    map_mask_plot[:, xind, yind] = np.array(MapUtils.colors[layer])[:, None]
                map_mask_plot = np.transpose(map_mask_plot, (2,1,0))
                map_mask_plot = np.flip(map_mask_plot,0)

                plt.imshow(map_mask_plot, alpha=0.8, extent=[plt.xlim()[0], plt.xlim()[1], plt.ylim()[0], plt.ylim()[1]])

                plt.savefig(imname)
                plt.close(fig)

                filenames.append(imname)

            if len(data['valid_id']) > 1: # one per scene
                break

        if vis:
            # build gif
            if len(filenames) > 0:
                gifname = f'{vis_dir}/{k}.gif'
                with imageio.get_writer(gifname, mode='I') as writer:
                    for filename in filenames:
                        image = imageio.imread(filename)
                        writer.append_data(image)
                print('saving gif', gifname)

                # Remove files
                for filename in set(filenames):
                    os.remove(filename)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--adv_cfg', default='base')
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)

    parser.add_argument('--version', default='mini')
    parser.add_argument('--dataroot', default='../../nuscenes')
    parser.add_argument('--map_folder', default='../../nuscenes/mini')

    parser.add_argument('--adv', action='store_true', default=False)
    parser.add_argument('--drive', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--no_vis', action='store_true', default=False)


    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
    adv_cfg = AdvConfig(args.adv_cfg)
    adv_cfg.sample_k = cfg.sample_k
    adv_cfg.traj_scale = cfg.traj_scale

    if args.epochs is None:
        epochs = [cfg.get_last_epoch()]
    else:
        epochs = [int(x) for x in args.epochs.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    adv_cfg.device = device
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)

    cfg.adv_cfg = adv_cfg
    
    # torch.set_grad_enabled(False)
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    for epoch in epochs:
        prepare_seed(cfg.seed)
        """ model """
        if not args.cached:
            model_id = cfg.get('model_id', 'agentformer')
            model = model_dict[model_id](cfg)
            model.set_device(device)
            model.eval()
            if epoch > 0:
                cp_path = cfg.model_path % epoch
                print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
                model_cp = torch.load(cp_path, map_location='cpu')
                model.load_state_dict(model_cp['model_dict'], strict=False)

        """ save results and compute metrics """
        data_splits = [args.data_eval]

        for split in data_splits:  
            # generator = data_generator(cfg, log, split=split, phase='testing')
            save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
            args.results_dir = save_dir
            eval_dir = f'{save_dir}/samples'
            if not args.cached:
                if args.drive:
                    drive_model(args, cfg)
                elif args.eval:
                    eval_model(args, cfg)
                else:
                    viz_model(args, cfg)

            # remove eval folder to save disk space
            if args.cleanup:
                shutil.rmtree(save_dir)


