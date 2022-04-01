import numpy as np
import os
import sys

import torch
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from utils.attack_utils.dynamic_model import DynamicModel, DynamicModelConfig

from pdb import set_trace as st

def longitudal_mean_displacements(orig_motion, motion):
        # orig_s = orig_motion[1:,:,:] - orig_motion[:-1,:,:]
        # orig_s_unit = torch.nan_to_num(orig_s/orig_s.norm(dim=-1).unsqueeze(-1).tile(1,1,2))
        # d_motion = motion - orig_motion
        motion_s = motion[1:,:,:] - motion[:-1,:,:]
        motion_s = torch.nan_to_num(motion_s.norm(dim=-1))
        # return (orig_s_unit * d_motion[1:,:,:]).sum(dim=-1).mean()
        return motion_s.mean()

def lateral_mean_displacements(orig_motion, motion):
    orig_s = orig_motion[1:,:,:] - orig_motion[:-1,:,:]
    orig_s_unit = torch.nan_to_num(orig_s/orig_s.norm(dim=-1).unsqueeze(-1).tile(1,1,2))
    orig_s_lat_unit = torch.matmul(orig_s_unit, torch.tensor([[0.,1.],[-1.,0.]]).float().to(orig_s.device))
    d_motion = motion - orig_motion
    return (orig_s_lat_unit * d_motion[1:,:,:]).sum(dim=-1).mean()

def Augment_DO(trajs, lat=True, device=None, debug=False):
    cfg = DynamicModelConfig(
        device = device,
        debug = debug,
        sample = 5
    )

    dm = DynamicModel(trajs, cfg)
    orig_motion = trajs

    if debug:
        print('='*10, ('Augmenting lateral' if lat else 'Augmenting longitudal'), '='*10)

    optimizer = torch.optim.Adam([dm.dk], lr=1e-3)
    optimizer_dds = torch.optim.Adam([dm.dds], lr=1e-3)

    # reconstruct
    for i in range(100):
        motion, heading, reg_loss = dm.build_motion()
        
        motion_loss, traj_loss = reg_loss[0], reg_loss[2]
        motion_s = reg_loss[-1]
        cost = (motion_loss + traj_loss)
        # cost = motion_loss
        # cost = - traj_loss
        if dm.cfg.debug:
            print(f'{i} cost: {cost.item():.3f}   motion: {motion_loss.item():.3f}    traj: {traj_loss.item():.3f}')
        optimizer.zero_grad()
        optimizer_dds.zero_grad()
        cost.backward()
        optimizer.step()
        optimizer_dds.step()

    # generate augmentation
    for i in range(100):
        motion, heading, reg_loss = dm.build_motion(clip=True)
        # optimizer = torch.optim.Adam([dm.dk], lr=1e-4)
        # optimizer_dds = torch.optim.Adam([dm.dds], lr=1e-2)
        motion_loss, traj_loss = reg_loss[0], reg_loss[1]
        if lat:
            obj_loss = lateral_mean_displacements(orig_motion, motion)
        else:
            obj_loss = longitudal_mean_displacements(orig_motion, motion)
        cost =  traj_loss - obj_loss
        # cost = - traj_loss
        if dm.cfg.debug:
            print(f'{i} cost: {cost.item():.3f} motion: {motion_loss.item():.3f} traj: {traj_loss.item():.3f} obj: {obj_loss.item():.3f}')
        optimizer.zero_grad()
        optimizer_dds.zero_grad()
        cost.backward()
        optimizer.step()
        optimizer_dds.step()

    return motion.detach().cpu().transpose(0,1)