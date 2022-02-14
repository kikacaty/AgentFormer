import torch
import numpy as np
from pdb import set_trace as st


def motion_bending_loss(fut_motion, pred_motion, pre_motion):
    s_fut = fut_motion - pre_motion[:,-1,:].unsqueeze(1)
    s_pred = pred_motion - pre_motion[:,-1,:].unsqueeze(1)
    h_fut = torch.atan2(s_fut[:,1],s_fut[:,0])
    h_pred = torch.atan2(s_pred[:,1],s_pred[:,0])

    h_diff = h_pred - h_fut
    h_diff = (h_diff + np.pi) % (2*np.pi) - np.pi

    return  h_diff.mean()

def collision_loss(pre_motion, adv_id = 0):
    dist = (pre_motion- pre_motion[:,adv_id,:].unsqueeze(1).tile(1,pre_motion.size(1),1)).norm(dim=-1) + 1
    return (1/dist).mean()

def mean_distances(fut_motion, pred_motion):
    return (fut_motion - pred_motion)[:,1:,:].norm(dim=-1).mean()

def longitudal_mean_displacements(fut_motion, pred_motion, pre_motion):
    s_fut = fut_motion - torch.cat((pre_motion[:,-1,:].unsqueeze(1),fut_motion[:,:-1,:]), 1)
    s_fut_unit = s_fut/s_fut.norm(dim=-1).unsqueeze(-1).tile(1,1,2)
    s_pred = pred_motion - torch.cat((pre_motion[:,-1,:].unsqueeze(1),fut_motion[:,:-1,:]), 1)
    return (s_fut_unit * s_pred).sum(-1).mean()

def lateral_mean_displacements(fut_motion, pred_motion, pre_motion):
    s_fut = fut_motion - torch.cat((pre_motion[:,-1,:].unsqueeze(1),fut_motion[:,:-1,:]), 1)
    s_fut_unit = s_fut/s_fut.norm(dim=-1).unsqueeze(-1).tile(1,1,2)
    s_fut_lat_unit = torch.matmul(s_fut_unit, torch.tensor([[0.,1.],[-1.,0.]]).float().to(s_fut.device))
    s_pred = pred_motion - torch.cat((pre_motion[:,-1,:].unsqueeze(1),fut_motion[:,:-1,:]), 1)

    return (s_fut_lat_unit * s_pred).sum(-1).mean()

def final_distances(fut_motion, pred_motion):
    return (fut_motion - pred_motion).norm(dim=-1)[-1,:].mean()