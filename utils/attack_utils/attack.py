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

    # noise based perturbation
    def perturb_noise(self, data):
        model = self.model
        cfg = self.cfg
        device = cfg.device
        sample_k = cfg.sample_k
        fix_t = cfg.fix_t
        assert (fix_t == 0 or fix_t == -1)
        target_id = cfg.target_id


        model.set_data(data)

        orig_pre_motion = model.data['pre_motion'].clone().detach()

        dynamic_model = DynamicModel(orig_pre_motion.cpu().data[:,0,:], device=device, cfg = cfg)

        perturb_mask = torch.zeros_like(model.data['pre_motion'])
        # fixed current position
        perturb_mask[:,0,:] = 1

        for i in range(self.max_iters+1):

            adv_motion, heading, _ = dynamic_model.build_motion()
            update_adv_pre_motion = orig_pre_motion.clone()
            update_adv_pre_motion[:,0,:] = adv_motion
            model.data['pre_motion'] = update_adv_pre_motion
            model.data['heading'][0] = heading
            
            model.update_data(data)

            recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)

            sample_motion_3D, _ = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
            sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()

            # save results
            if i in cfg.iters:
                self.sample_motion_list.append(sample_motion_3D.detach().clone())
                self.recon_motion_list.append(recon_motion_3D.detach().clone())

            if target_id == -1:
                target_fut_motion = model.data['fut_motion'].transpose(0, 1)[1:,:,:]
                target_pred_motion = sample_motion_3D[0][1:,:,:]
                target_pre_motion = orig_pre_motion.detach().transpose(0, 1)[1:,:,:]
            else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                target_fut_motion = model.data['fut_motion'].transpose(0, 1)[target_id,:,:].unsqueeze(0)
                target_pred_motion = sample_motion_3D[0][target_id].unsqueeze(0)
                target_pre_motion = orig_pre_motion.detach().transpose(0, 1)[target_id,:,:].unsqueeze(0)
            # target_pred_motion = recon_motion_3D.contiguous()[1:]
            model.zero_grad()

            loss_bending = motion_bending_loss(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_mean_distance = mean_distances(target_fut_motion, target_pred_motion).to(device)
            loss_lon = longitudal_mean_displacements(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_lat = lateral_mean_displacements(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_col = collision_loss(update_adv_pre_motion, adv_id=0)

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

    # searching constraints
    def perturb_search(self, data):
        model = self.model
        cfg = self.cfg
        device = cfg.device
        sample_k = cfg.sample_k
        fix_t = cfg.fix_t
        assert (fix_t == 0 or fix_t == -1)
        target_id = cfg.target_id


        model.set_data(data)

        orig_pre_motion = model.data['pre_motion'].clone().detach()

        dynamic_model = DynamicModel(orig_pre_motion.cpu().data[:,0,:], device=device, cfg = cfg, constrained=True)

        perturb_mask = torch.zeros_like(model.data['pre_motion'])
        # fixed current position
        perturb_mask[:,0,:] = 1

        for i in range(self.max_iters+1):

            adv_motion, heading, _ = dynamic_model.build_motion()
            update_adv_pre_motion = orig_pre_motion.clone()
            update_adv_pre_motion[:,0,:] = adv_motion
            model.data['pre_motion'] = update_adv_pre_motion
            model.data['heading'][0] = heading
            
            model.update_data(data)

            recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)

            sample_motion_3D, _ = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
            sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()

            # save results
            if i in cfg.iters:
                self.sample_motion_list.append(sample_motion_3D.detach().clone())
                self.recon_motion_list.append(recon_motion_3D.detach().clone())

            if target_id == -1:
                target_fut_motion = model.data['fut_motion'].transpose(0, 1)[1:,:,:]
                target_pred_motion = sample_motion_3D[0][1:,:,:]
                target_pre_motion = orig_pre_motion.detach().transpose(0, 1)[1:,:,:]
            else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                target_fut_motion = model.data['fut_motion'].transpose(0, 1)[target_id,:,:].unsqueeze(0)
                target_pred_motion = sample_motion_3D[0][target_id].unsqueeze(0)
                target_pre_motion = orig_pre_motion.detach().transpose(0, 1)[target_id,:,:].unsqueeze(0)
            # target_pred_motion = recon_motion_3D.contiguous()[1:]
            model.zero_grad()

            loss_bending = motion_bending_loss(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_mean_distance = mean_distances(target_fut_motion, target_pred_motion).to(device)
            loss_lon = longitudal_mean_displacements(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_lat = lateral_mean_displacements(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_col = collision_loss(update_adv_pre_motion, adv_id=0)

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

    # dynamics optimization
    def perturb_opt(self, data):
        model = self.model
        cfg = self.cfg
        device = cfg.device
        sample_k = cfg.sample_k
        fix_t = cfg.fix_t
        assert (fix_t == 0 or fix_t == -1)
        target_id = cfg.target_id


        model.set_data(data)

        orig_pre_motion = model.data['pre_motion'].clone().detach()

        dynamic_model = DynamicModelOpt(orig_pre_motion.cpu().data[:,0,:], device=device, cfg = cfg)

        perturb_mask = torch.zeros_like(model.data['pre_motion'])
        # fixed current position
        perturb_mask[:,0,:] = 1

        for i in range(self.max_iters+1):

            adv_motion, heading, loss_motion, loss_traj = dynamic_model.build_motion()
            update_adv_pre_motion = orig_pre_motion.clone()
            update_adv_pre_motion[:,0,:] = update_adv_pre_motion[fix_t,0,:] + (adv_motion - adv_motion[fix_t,:])
            model.data['pre_motion'] = update_adv_pre_motion
            model.data['heading'][0] = heading
            
            model.update_data(data)

            recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)

            sample_motion_3D, _ = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
            sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()

            # save results
            if i in cfg.iters:
                self.sample_motion_list.append(sample_motion_3D.detach().clone())
                self.recon_motion_list.append(recon_motion_3D.detach().clone())

            if target_id == -1:
                target_fut_motion = model.data['fut_motion'].transpose(0, 1)[1:,:,:]
                target_pred_motion = sample_motion_3D[0][1:,:,:]
                target_pre_motion = orig_pre_motion.detach().transpose(0, 1)[1:,:,:]
            else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                target_fut_motion = model.data['fut_motion'].transpose(0, 1)[target_id,:,:].unsqueeze(0)
                target_pred_motion = sample_motion_3D[0][target_id].unsqueeze(0)
                target_pre_motion = orig_pre_motion.detach().transpose(0, 1)[target_id,:,:].unsqueeze(0)
            # target_pred_motion = recon_motion_3D.contiguous()[1:]
            model.zero_grad()

            loss_bending = motion_bending_loss(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_mean_distance = mean_distances(target_fut_motion, target_pred_motion).to(device)
            loss_lon = longitudal_mean_displacements(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_lat = lateral_mean_displacements(target_fut_motion, target_pred_motion, target_pre_motion)
            loss_col = collision_loss(update_adv_pre_motion, adv_id=0)

            loss_dict = {
                "ADE": loss_mean_distance.item(),
                "Longitudinal": loss_lon.item(),
                "Lateral": loss_lat.item(),
                "colission": loss_col.item(),
                "Motion": loss_motion.item(),
                "Pre Traj": loss_traj.item(),
            }

            cost = loss_mean_distance + (loss_motion + cfg.traj_reg * loss_traj) * cfg.motion_reg + loss_col * cfg.collision_reg
            if cfg.debug:
                print(f'Iter: {i:3d} \t Loss: {cost.item():3f}, ('+ 
                    ' '.join([f'{loss_name}: {loss_val:3f}' for loss_name, loss_val in loss_dict.items()])+
                    ')')
            cost.backward()

            dynamic_model.update_control()

        model.data['pre_motion'].detach_()