import numpy as np
import torch

from dataclasses import dataclass
from matplotlib import pyplot as plt

from pdb import set_trace as st

""" Dynamic Bicycle Model """
# TODO: update with a torch version
from scipy.interpolate import interp1d
def xy2spline(xy, ts):

    t = ts

    return interp1d(t, xy, kind='previous', axis=0, copy=False,
                    bounds_error=True, assume_sorted=True)

@dataclass
class DynamicModelConfig:
    fix_t : int = 0
    device : torch.device = None
    sample : int = 5
    dt : float = 0.5
    debug : bool = False

class DynamicModel(object):
    def __init__(self, xy, cfg):
        self.bicycle_params = {
            'k': 0.2, # curvature 
            'dk': 0.05,
            'dh': np.pi/6,
            'ddh': np.pi/10,
            'a_s': 2, # acc
            'a_s_lb': -4, # acc
            'a_l': 3, # lateral acc
            'ds': 15 # assume the speed is constrained
        }

        self.device = cfg.device
        self.xy = xy.clone().to(self.device)
        self.cfg = cfg
        dt = cfg.dt
        sample = cfg.sample

        self.ts_s = np.arange(len(xy)*sample - sample + 1) * (dt/sample)
        self.ts = np.arange(len(xy)) * dt
        self.idx = np.arange(0, len(xy)*sample, sample) 
        self.dt = dt/sample
        self.init_dt = dt
        self.fix_t = cfg.fix_t

        self.spline = xy2spline(self.xy.cpu(),self.ts)

        if cfg.debug:
            torch.set_printoptions(sci_mode=False, precision=3)

        self.init_dynamics()

    def d(self, x, is_heading=False):
        dx = torch.zeros_like(x)
        dx[1:] = x[1:] - x[:-1]
        dx[0] = dx[1] # assume to be the same
        if is_heading:
            dx = (dx + np.pi) % (2 * np.pi) - np.pi
        return dx/self.init_dt # scaling t

    def save_recon(self):
        self.dk_recon = self.dk.detach().clone()
        self.dds_recon = self.dds.detach().clone()

    def load_recon(self):
        self.dk = self.dk_recon.clone()
        self.dds = self.dds.clone()

    def init_dynamics(self):

        d = self.d
        xy = self.xy.cpu()

        dx = d(xy[:,:,0])[...,None]
        dy = d(xy[:,:,1])[...,None]
        
        ddx = d(dx)
        ddy = d(dy)

        ds = torch.sqrt(dx**2 + dy**2)
        dds = d(ds)

        h = torch.atan2(dy,dx)
        for agent_idx in range(h.shape[1]):
            for f_idx in range(h.shape[0]-1):
                if ds[f_idx+1,agent_idx] == 0: # if speed is 0, remain the previous heading
                    h[f_idx+1,agent_idx] = h[f_idx,agent_idx]

        dh = d(h, is_heading=True)

        ddh = d(dh)

        k = dh/ds
        k[ds==0] = 0
        dk = d(k)

        a_l = k * (ds ** 2)

        new_dk = torch.from_numpy(xy2spline(dk, self.ts)(self.ts_s))
        new_dds = torch.from_numpy(xy2spline(dds, self.ts)(self.ts_s))

        new_k = k[0] + self.dt * (new_dk.cumsum(0) - new_dk.cumsum(0)[0])

        new_ds = ds[0] + self.dt * (new_dds.cumsum(0) - new_dds.cumsum(0)[0])

        new_dh = new_k * new_ds

        new_h = h[0] + self.dt * (new_dh.cumsum(0) - new_dh.cumsum(0)[0])
        
        new_vel = torch.cat([torch.cos(new_h) * new_ds, torch.sin(new_h) * new_ds], dim=-1)
        new_xy_s = xy[0, :] + self.dt * (new_vel.cumsum(0)- new_vel.cumsum(0)[0])

        self.ds = new_ds.float().to(self.device)
        self.dds = new_dds.float().to(self.device)
        self.dh = new_dh.float().to(self.device)
        self.k = new_k.float().to(self.device)
        self.dk = new_dk.float().to(self.device)

        self.h = new_h.float().to(self.device)
        self.orig_xy_s = new_xy_s.float().to(self.device)

        # rand init
        # self.dds += torch.randn_like(self.dds) * 1e-3
        # self.dk += torch.randn_like(self.dk) * 1e-3

    def get_violations(self):
        k_violations = (torch.abs(self.k.detach().cpu().numpy()) > self.bicycle_params['k']).mean()
        dds_violations = (torch.abs(self.dds.detach().cpu().numpy()) > self.bicycle_params['a_s']).mean()

        return k_violations,dds_violations


    def build_motion(self, clip=False):

        self.dds.requires_grad = True
        self.dk.requires_grad = True

        new_k = self.k.clone().detach()
        new_k = new_k[self.fix_t] + self.dt * (self.dk.cumsum(0) - self.dk.cumsum(0)[self.fix_t])
        if clip:
            new_k = torch.clip(new_k, min=-self.bicycle_params['k'], max=self.bicycle_params['k'])

        new_ds = self.ds.clone().detach()
        new_ds = new_ds[self.fix_t] + self.dt * (self.dds.cumsum(0) - self.dds.cumsum(0)[self.fix_t])
        if clip:
            new_ds = torch.clip(new_ds, min=-self.bicycle_params['ds'], max=self.bicycle_params['ds'])


        new_dh = new_k * new_ds
        if clip:
            new_dh = torch.clip(new_dh, min=-self.bicycle_params['dh'], max=self.bicycle_params['dh'])

        new_h = self.h.clone().detach()
        new_h = new_h[self.fix_t] + self.dt * (new_dh.cumsum(0) - new_dh.cumsum(0)[self.fix_t])
        
        new_vel = torch.cat([torch.cos(new_h) * new_ds, torch.sin(new_h) * new_ds], dim=-1)
        new_xy_s = self.orig_xy_s.clone().detach()
        new_xy_s = new_xy_s[self.fix_t, :] + self.dt * (new_vel.cumsum(0)- new_vel.cumsum(0)[self.fix_t])

        bound_func = lambda x: x - torch.sigmoid(x) + 0.5
        bound_func = lambda x: torch.exp(torch.clip(x,min=0.9,max=5)-0.9)
        
        loss_mask = torch.abs(new_ds) > 0.1
        loss_motion = torch.stack((bound_func(torch.abs(self.dds[loss_mask] - self.bicycle_params['a_s_lb'])/(self.bicycle_params['a_s'] - self.bicycle_params['a_s_lb'])).mean(),
                        bound_func(torch.abs(self.dk[loss_mask])/self.bicycle_params['dk']).mean(),
                        bound_func(torch.abs(self.k[loss_mask])/self.bicycle_params['k']).mean(),
                        torch.abs(self.dk[1:]-self.dk[:-1]).mean(),
                        torch.abs(self.dds[1:]-self.dds[:-1]).mean(),
                    ))

        loss_motion = torch.mean(loss_motion)

        loss_traj =  bound_func((new_xy_s[self.idx] - self.xy).norm(dim=-1)/1).mean()

        loss_traj_init =  ((new_xy_s[self.idx] - self.xy).norm(dim=-1)/1).mean()

        new_xy = new_xy_s[self.idx]

        loss_intention = ((new_xy[1:] - new_xy[:-1]) - (self.xy[1:] - self.xy[:-1])).norm(dim=-1).mean()

        heading = new_h.squeeze(-1)[-1]

        return new_xy, heading, [loss_motion, loss_traj, loss_traj_init, loss_intention, new_xy_s]
