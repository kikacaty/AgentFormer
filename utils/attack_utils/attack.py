'''
File: attack.py
File Created: Saturday, 5th February 2022 3:28:06 pm
-----
This code is borrowed from  Yulong Cao, 2022
-----
'''

import torch
from torch.optim import lr_scheduler
from torch import nn

import numpy as np

from .loss import *
from .constraint import *

from pdb import set_trace as st

class Attacker(object):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.max_iters = np.max(cfg.iters)
        self.sample_motion_list = []
        self.recon_motion_list = []

    # inference
    def perturb(self, data):
        model = self.model
        cfg = self.cfg
        device = cfg.device
        sample_k = cfg.sample_k
        fix_t = cfg.fix_t.t_idx
        target_agent = cfg.target_agent
        adv_agent = cfg.adv_agent


        model.set_data(data)

        model.inference(mode='infer', sample_num=sample_k, need_weights=False)

        orig_q_z_dist = model.data['q_z_dist_dlow'].copy()

        orig_pre_motion = model.data['pre_motion'].clone().detach() # frame, na, xy

        if cfg.mode == 'opt':
            dynamic_model = DynamicModelOpt(orig_pre_motion.data, device=device, cfg = cfg)
        else:
            dynamic_model = DynamicModel(orig_pre_motion.data, device=device, cfg = cfg, constrained=(cfg.mode=='search'))

        if adv_agent.all:
            adv_motion_mask = torch.ones_like(model.data['pre_motion'])
            adv_heading_mask = torch.ones_like(model.data['heading'])
        else:
            adv_motion_mask = torch.zeros_like(model.data['pre_motion'])
            # fixed current position
            adv_motion_mask[:,adv_agent.idx,:] = 1

            adv_heading_mask = torch.zeros_like(model.data['heading'])
            # fixed current position
            adv_heading_mask[adv_agent.idx] = 1

        for i in range(self.max_iters+1):

            adv_motion, adv_heading, reg_loss = dynamic_model.build_motion()
            update_adv_pre_motion = orig_pre_motion.clone()
            if cfg.mode == 'opt':
                adv_motion = update_adv_pre_motion[fix_t] + (adv_motion - adv_motion[fix_t])

            headings = model.data['heading'].clone().detach()

            update_adv_pre_motion = update_adv_pre_motion * (1-adv_motion_mask) + adv_motion_mask * adv_motion
            headings = headings * (1 - adv_heading_mask) + adv_heading_mask * adv_heading

            model.update_data(data, pre_motion=update_adv_pre_motion, heading=headings)

            recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)

            sample_motion_3D, _ = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
            sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()

            # save results
            if i in cfg.iters:
                self.sample_motion_list.append(sample_motion_3D.detach().clone())
                self.recon_motion_list.append(recon_motion_3D.detach().clone())

            if target_agent.all:
                target_fut_motion = model.data['fut_motion'].transpose(0, 1)
                target_pred_motion = sample_motion_3D[0]
                target_pred_motion_samples = sample_motion_3D.transpose(0, 1)
                target_pre_motion = orig_pre_motion.detach().transpose(0, 1)
                if target_agent.other:
                    target_fut_motion = target_fut_motion[torch.arange(target_fut_motion.size(0))!=adv_agent.idx]
                    target_pred_motion = target_pred_motion[torch.arange(target_pred_motion.size(0))!=adv_agent.idx]
                    target_pred_motion_samples = target_pred_motion_samples[torch.arange(sample_motion_3D.size(1))!=adv_agent.idx]
                    target_pre_motion = target_pre_motion[torch.arange(target_pre_motion.size(0))!=adv_agent.idx]
            else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                target_fut_motion = model.data['fut_motion'].transpose(0, 1)[target_agent.idx].unsqueeze(0)
                target_pred_motion = sample_motion_3D[0][target_agent.idx].unsqueeze(0)
                target_pred_motion_samples = sample_motion_3D.transpose(0, 1)[target_agent.idx].unsqueeze(0)
                target_pre_motion = orig_pre_motion.detach().transpose(0, 1)[target_agent.idx].unsqueeze(0)
            # target_pred_motion = recon_motion_3D.contiguous()[1:]
            model.zero_grad()

            loss_min_mean_distance = min_mean_distances(target_fut_motion, target_pred_motion_samples).to(device)

            

            loss_bending = motion_bending_loss(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_mean_distance = mean_distances(target_fut_motion, target_pred_motion).to(device)
            loss_lon = longitudal_mean_displacements(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_lat = lateral_mean_displacements(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_col = collision_loss(update_adv_pre_motion, adv_id=0)

            if cfg.mode == 'opt':
                loss_motion, loss_traj, _ = reg_loss
                loss_dict = {
                    "ADE": loss_mean_distance.item(),
                    "Longitudinal": loss_lon.item(),
                    "Lateral": loss_lat.item(),
                    "colission": loss_col.item(),
                    "Motion": loss_motion.item(),
                    "Pre Traj": loss_traj.item(),
                }

                cost = loss_min_mean_distance + (loss_motion + cfg.traj_reg * loss_traj) * cfg.motion_reg + loss_col * cfg.collision_reg
            else:
                loss_dict = {
                    "ADE": loss_mean_distance.item(),
                    "Longitudinal": loss_lon.item(),
                    "Lateral": loss_lat.item(),
                    "colission": loss_col.item(),
                }

                cost = loss_mean_distance + loss_col * cfg.collision_reg
            if cfg.debug:
                print(f'Iter: {i:3d} \t Loss: {cost.item():3f}, ('+ 
                    ' '.join([f'{loss_name}: {loss_val:3f}' for loss_name, loss_val in loss_dict.items()])+
                    ')')
            cost.backward()

            dynamic_model.update_control()

        model.data['pre_motion'].detach_()
        model.data['heading'].detach_()
    
    # training
    # noise
    def perturb_train(self, data):
        model = self.model
        cfg = self.cfg
        device = cfg.device
        fix_t = cfg.fix_t.t_idx
        target_agent = cfg.target_agent
        adv_agent = cfg.adv_agent


        model.set_data(data)

        orig_pre_motion = model.data['pre_motion'].clone().detach() # frame, na, xy

        if cfg.mode == 'opt':
            dynamic_model = DynamicModelOpt(orig_pre_motion.data, device=device, cfg = cfg)
        else:
            dynamic_model = DynamicModel(orig_pre_motion.data, device=device, cfg = cfg, constrained=(cfg.mode=='search'))

        if adv_agent.all:
            adv_motion_mask = torch.ones_like(model.data['pre_motion'])
            adv_heading_mask = torch.ones_like(model.data['heading'])
        else:
            adv_motion_mask = torch.zeros_like(model.data['pre_motion'])
            # fixed current position
            adv_motion_mask[:,adv_agent.idx,:] = 1

            adv_heading_mask = torch.zeros_like(model.data['heading'])
            # fixed current position
            adv_heading_mask[adv_agent.idx] = 1

        for i in range(self.max_iters+1):

            adv_motion, adv_heading, reg_loss = dynamic_model.build_motion()
            update_adv_pre_motion = orig_pre_motion.clone()
            if cfg.mode == 'opt':
                adv_motion = update_adv_pre_motion[fix_t] + (adv_motion - adv_motion[fix_t])
            
            headings = model.data['heading'].clone().detach()
            
            update_adv_pre_motion = update_adv_pre_motion * (1-adv_motion_mask) + adv_motion_mask * adv_motion
            headings = headings * (1 - adv_heading_mask) + adv_heading_mask * adv_heading

            model.update_data(data, pre_motion = update_adv_pre_motion, heading = headings)

            recon_motion_3D, _ = model.inference(mode='recon')

            if target_agent.all:
                target_fut_motion = model.data['fut_motion'].transpose(0, 1)
                target_pred_motion = recon_motion_3D
                target_pre_motion = orig_pre_motion.detach().transpose(0, 1)
                if target_agent.other:
                    target_fut_motion = target_fut_motion[torch.arange(target_fut_motion.size(0))!=adv_agent.idx]
                    target_pred_motion = target_pred_motion[torch.arange(target_pred_motion.size(0))!=adv_agent.idx]
                    target_pre_motion = target_pre_motion[torch.arange(target_pre_motion.size(0))!=adv_agent.idx]
            else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                target_fut_motion = model.data['fut_motion'].transpose(0, 1)[target_agent.idx].unsqueeze(0)
                target_pred_motion = recon_motion_3D[target_agent.idx].unsqueeze(0)
                target_pre_motion = orig_pre_motion.detach().transpose(0, 1)[target_agent.idx].unsqueeze(0)
            # target_pred_motion = recon_motion_3D.contiguous()[1:]
            model.zero_grad()

            loss_bending = motion_bending_loss(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_mean_distance = mean_distances(target_fut_motion, target_pred_motion).to(device)
            loss_lon = longitudal_mean_displacements(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_lat = lateral_mean_displacements(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_col = collision_loss(update_adv_pre_motion, adv_id=0)

            if cfg.mode == 'opt':
                loss_motion, loss_traj, _ = reg_loss
                loss_dict = {
                    "ADE": loss_mean_distance.item(),
                    "Longitudinal": loss_lon.item(),
                    "Lateral": loss_lat.item(),
                    "colission": loss_col.item(),
                    "Motion": loss_motion.item(),
                    "Pre Traj": loss_traj.item(),
                }

                cost = loss_mean_distance + (loss_motion + cfg.traj_reg * loss_traj) * cfg.motion_reg + loss_col * cfg.collision_reg
            else:
                loss_dict = {
                    "ADE": loss_mean_distance.item(),
                    "Longitudinal": loss_lon.item(),
                    "Lateral": loss_lat.item(),
                    "colission": loss_col.item(),
                }

                cost = loss_mean_distance + loss_col * cfg.collision_reg
            if cfg.debug:
                print(f'Iter: {i:3d} \t Loss: {cost.item():3f}, ('+ 
                    ' '.join([f'{loss_name}: {loss_val:3f}' for loss_name, loss_val in loss_dict.items()])+
                    ')')
            cost.backward()

            dynamic_model.update_control()

        # print(dynamic_model.perturbation)

        model.data['pre_motion'].detach_()
        model.data['heading'].detach_()
        model.update_data(data, pre_motion = model.data['pre_motion'], heading = model.data['heading'])