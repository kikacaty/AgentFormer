import numpy as np
import torch

from matplotlib import pyplot as plt

from pdb import set_trace as st

""" Dynamic Bicycle Model """
# TODO: update with a torch version
from scipy.interpolate import interp1d
def xy2spline(xy, ts):

    t = ts

    return interp1d(t, xy, kind='linear', axis=0, copy=False,
                    bounds_error=True, assume_sorted=True)

class DynamicModelOpt(object):
    def __init__(self, xy, dt=0.5, sample=5, device=None, cfg=None, use_k=True):
        self.bicycle_params = {
            'k': 0.2, # curvature 
            'dk': 0.05,
            'dh': np.pi/6,
            'ddh': np.pi/10,
            'a_s': 2, # acc
            'a_l': 3, # lateral acc
            'ds': 15 # assume the speed is constrained
        }
        
        self.xy = xy.clone() * cfg.traj_scale
        self.device = device
        self.cfg = cfg
        self.use_k = use_k

        self.ts_s = np.arange(len(xy)*sample - sample + 1) * (dt/sample)
        self.ts = np.arange(len(xy)) * dt
        self.idx = np.arange(0, len(xy)*sample, sample) 
        self.dt = dt/sample
        self.fix_t = cfg.fix_t

        self.spline = xy2spline(self.xy,self.ts)

        self.init_states = self.init_dynamics()

        # # self check
        # xy = self.perturb(0, vel=False)
        # states = self.get_states(xy)
        # for key in self.bicycle_params.keys():
        #     if (np.abs(states[key]) > self.bicycle_params[key]).sum() != 0: # check all ts satisfy constraints
        #         print("========== Original traj failed dynamic check. ==============")
        #         print(states)
        #         break

    def init_dynamics(self):
        def d(x):
            dx = np.zeros_like(x)
            dx[:-1] = x[1:] - x[:-1]
            dx[-1] = dx[-2] # assume to be the same
            return dx/self.dt # scaling t

        xy = self.spline(self.ts_s)

        dx = d(xy[:,0])
        dy = d(xy[:,1])
        
        ddx = d(dx)
        ddy = d(dy)

        h = np.arctan2(dy,dx)
        dh = d(h)
        # dh_list = [dh + np.pi, dh, dh - np.pi]
        # dh = np.choose(np.argmin(np.abs(dh_list),axis=0), dh_list) 
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        ddh = d(dh)

        ds = np.sqrt(dx**2 + dy**2)
        dds = d(ds)

        k = dh/ds
        k[ds==0] = 0
        dk = d(k)

        a_l = k * (ds ** 2)

        self.dds_orig = torch.tensor(dds).float().to(self.device)
        self.dh_orig = torch.tensor(dh).float().to(self.device)

        self.dds = torch.tensor(dds).float().to(self.device)
        self.dh = torch.tensor(dh).float().to(self.device)
        self.k = torch.tensor(k).float().to(self.device)
        self.ds = torch.tensor(ds).float().to(self.device)

        self.h = torch.tensor(h).float().to(self.device)
        self.xy_s = torch.tensor(xy).float().to(self.device)

        return {
            'k': k, # curvature 
            'dk': dk,
            'ds': ds,
            'a_s': dds, # acc
            'a_l': a_l, # lateral acc
            'dh': dh,
            'ddh': ddh,
        }

    def build_motion(self):

        use_k = self.use_k

        self.dds.requires_grad = True
        if use_k: 
            self.k.requires_grad = True
        else:
            self.dh.requires_grad = True


        new_ds = self.ds.clone().detach()
        new_ds = new_ds[self.fix_t] + self.dt * (self.dds.cumsum(0) - self.dds.cumsum(0)[self.fix_t])

        if use_k:
            new_dh = self.k * new_ds
        else:
            new_dh = self.dh

        new_h = self.h.clone().detach()
        new_h = new_h[self.fix_t] + self.dt * (new_dh.cumsum(0) - new_dh.cumsum(0)[self.fix_t])

        
        new_ds = new_ds.unsqueeze(-1)
        new_h = new_h.unsqueeze(-1)
        
        new_vel = torch.cat([torch.cos(new_h) * new_ds, torch.sin(new_h) * new_ds], dim=-1)
        new_xy_s = self.xy_s.clone().detach()
        new_xy_s = new_xy_s[self.fix_t, :] + self.dt * (new_vel.cumsum(0)- new_vel.cumsum(0)[self.fix_t])

        bound_func = lambda x: x - torch.sigmoid(x) + 0.5

        loss_motion = - bound_func(torch.abs(self.dh)/self.bicycle_params['dh']).mean() \
                        - bound_func(torch.abs(self.dds)/self.bicycle_params['a_s']).mean() \
                        - bound_func(torch.abs(self.ds)/self.bicycle_params['ds']).mean() \
                        - bound_func(torch.abs(self.k)/self.bicycle_params['k']).mean()

        loss_motion = loss_motion / 4

        loss_traj =  - bound_func((new_xy_s - self.xy_s).norm(dim=-1)/1).mean()
        loss_head =  - (new_h - self.h).norm().mean()

        new_xy = new_xy_s[self.idx]/self.cfg.traj_scale

        heading = new_h[-1]

        return new_xy, heading, loss_motion, loss_traj

    def update_control(self):

        use_k = self.use_k
        noise = self.cfg.noise

        if noise > 0:
            adv_dds = self.dds + torch.randn_like(self.dds) * self.cfg.step_size_dds * self.bicycle_params['a_s'] * noise
            adv_dh = self.dh + torch.randn_like(self.dh) * self.cfg.step_size_dh * self.bicycle_params['dh'] * noise
            adv_k = self.k + torch.randn_like(self.k) * self.cfg.step_size_dk * self.bicycle_params['k'] * noise
        else:
            adv_dds = self.dds
            adv_dh = self.dh
            adv_k = self.k

        adv_dds += self.cfg.step_size_dds*self.bicycle_params['a_s']*self.dds.grad.sign()
        # self.dds = adv_dds.detach_()
        self.dds = torch.clamp(adv_dds, min=-2*self.bicycle_params['a_s'], max=self.bicycle_params['a_s']).detach_()
        if not use_k:   
            adv_dh += self.cfg.step_size_dh*self.bicycle_params['dh']*self.dh.grad.sign()
            # self.dh = adv_dh.detach_()
            self.dh = torch.clamp(adv_dh, min=-self.bicycle_params['dh'], max=self.bicycle_params['dh']).detach_() 
        else:
            adv_k += self.cfg.step_size_dk*self.bicycle_params['k']*self.k.grad.sign()
            # self.k = adv_k.detach_()
            self.k = torch.clamp(adv_k, min=-self.bicycle_params['k'], max=self.bicycle_params['k']).detach_() 


class DynamicModel(object):
    def __init__(self, xy, dt=0.5, sample=1, device=None, cfg=None, use_k=True, constrained=False):
        self.bicycle_params = {
            'k': 0.2, # curvature 
            'dk': 0.05,
            'dh': np.pi/6,
            'ddh': np.pi/10,
            'a_s': 2, # acc
            # 'a_l': 3, # lateral acc
            'ds': 15 # assume the speed is constrained
        }
        
        self.xy = xy.clone().to(device) * cfg.traj_scale
        self.device = device
        self.cfg = cfg
        self.use_k = use_k
        self.constrained = constrained

        self.ts_s = np.arange(len(xy)*sample - sample + 1) * (dt/sample)
        self.ts = np.arange(len(xy)) * dt
        self.idx = np.arange(0, len(xy)*sample, sample) 
        self.dt = dt/sample
        self.fix_t = cfg.fix_t
        self.eps = cfg.eps
        self.step_size = cfg.step_size
        self.traj_scale =cfg.traj_scale

        self.init_states = self.init_dynamics()

        # # self check
        # xy = self.perturb(0, vel=False)
        # states = self.get_states(xy)
        # for key in self.bicycle_params.keys():
        #     if (np.abs(states[key]) > self.bicycle_params[key]).sum() != 0: # check all ts satisfy constraints
        #         print("========== Original traj failed dynamic check. ==============")
        #         print(states)
        #         break

    def init_dynamics(self):
        self.perturbation = torch.zeros_like(self.xy).to(self.device)
        return self.get_dynamics(self.xy)

    def get_dynamics(self, xy):
        def d(x):
            dx = torch.zeros_like(x)
            dx[:-1] = x[1:] - x[:-1]
            dx[-1] = dx[-2] # assume to be the same
            return dx/self.dt # scaling t

        dx = d(xy[:,0])
        dy = d(xy[:,1])
        
        ddx = d(dx)
        ddy = d(dy)

        h = torch.atan2(dy,dx)
        dh = d(h)
        dh = (dh + np.pi) % (2*np.pi) - np.pi
        ddh = d(dh)

        ds = torch.sqrt(dx**2 + dy**2)
        a_s = dds = d(ds)

        k = dh/ds
        k[ds==0] = 0
        dk = d(k)

        a_l = k * (ds ** 2)

        return {
            'k': k, # curvature 
            'dk': dk,
            'ds': ds,
            'a_s': a_s, # acc
            'a_l': a_l, # lateral acc
            'h': h,
            'dh': dh,
            'ddh': ddh,
        }

    def build_motion(self):

        xy = self.xy.clone().detach().to(self.device)

        self.perturbation.requires_grad = True
        new_xy = xy + self.perturbation

        states = self.get_dynamics(new_xy)

        heading = states['h'][-1]

        return new_xy/self.traj_scale, heading, []

    def update_control(self):

        if self.cfg.noise > 0:
            self.perturbation = torch.clamp(self.perturbation + self.cfg.noise*self.step_size*torch.randn_like(self.perturbation) + self.step_size*self.perturbation.grad.sign(), min=-self.eps, max=self.eps).detach_()
        else:
            self.perturbation = torch.clamp(self.perturbation + self.step_size*torch.randn_like(self.perturbation) + self.step_size*self.perturbation.grad.sign(), min=-self.eps, max=self.eps).detach_()

        if self.constrained:
            self.update_constraints()

    def update_constraints(self):

        gamma = 0.01
        scale = 1.0 + gamma
        constrained = 0
        xy = self.xy.clone().detach()
        perturbation = self.perturbation.clone().detach()
        while constrained < len(self.bicycle_params.keys()) and scale > 0:
            constrained = 0
            scale -= gamma

            cur_perturbation = scale * perturbation
            c_xy = xy + cur_perturbation
            states = self.get_dynamics(c_xy)

            for key in self.bicycle_params.keys():
                if (torch.abs(states[key]) > self.bicycle_params[key]).sum() != 0: # check all ts satisfy constraints
                    break
                constrained += 1
        
        self.perturbation = (scale * perturbation).detach().to(self.device)

class DynamicModelSearch(object):
    def __init__(self, xy, dt=0.5):
        self.bicycle_params = {
            'k': 0.2, # curvature 
            'dk': 0.05,
            'dh': np.pi/6,
            'ddh': np.pi/10,
            'a_s': 5, # acc
            'a_l': 3, # lateral acc
            # 'ds': 15 # assume the speed is constrained
        }
        
        self.states = {}
        self.xy = xy.clone()

        self.ts = (np.arange(len(xy)) - len(xy) + 1) * dt
        self.dt = dt

        # self check
        xy = self.perturb(0, vel=False)
        states = self.get_states(xy)
        for key in self.bicycle_params.keys():
            if (np.abs(states[key]) > self.bicycle_params[key]).sum() != 0: # check all ts satisfy constraints
                print("========== Original traj failed dynamic check. ==============")
                print(states)
                break

    def get_states(self, xy):
        def d(x):
            dx = x[1:] - x[:-1]
            return dx/self.dt # scaling t

        dx = d(xy[:,0])
        dy = d(xy[:,1])
        
        ddx = d(dx)
        ddy = d(dy)

        h = np.arctan2(dy,dx)
        dh = d(h)
        ddh = d(dh)

        # mask_h = (dx * dx + dy * dy == 0) # static
        # dh[mask_h] = 0
        # ddh[mask_h] = 0


        ds = np.sqrt(dx**2 + dy**2)
        dds = d(ds)

        # mask_k = (dx * dx + dy * dy <= 5**2)
        # k = (np.abs(ddx * dy - dx * ddy) / (dx * dx + dy * dy)**1.5)
        # k[mask_k] = 0 # curvature = 0 if speed < 5
        # dk = d(k)

        k = dh/ds[1:]
        k[ds[1:]==0] = 0
        dk = d(k)

        a_l = k * (ds[1:] ** 2)

        return {
            'k': k, # curvature 
            'dk': dk,
            'ds': ds,
            'a_s': dds, # acc
            'a_l': a_l, # lateral acc
            'dh': dh,
            'ddh': ddh,
        }

    def perturb(self, perturbation, vel=True):
        if vel:
            xy = self.xy.clone().detach()
            xy[:-1] = xy[-1] - (perturbation.sum(0) - perturbation.cumsum(0) + perturbation)
        else:
            xy = self.xy + perturbation
        return xy


    def apply_constraints(self, perturbation, vel = False):

        gamma = 0.01
        scale = 1.0 + gamma
        constrained = 0
        while constrained < len(self.bicycle_params.keys()) and scale > 0:
            constrained = 0
            scale -= gamma

            cur_perturbation = scale * perturbation
            xy = self.perturb(cur_perturbation, vel)
            states = self.get_states(xy)

            for key in self.bicycle_params.keys():
                if (np.abs(states[key]) > self.bicycle_params[key]).sum() != 0: # check all ts satisfy constraints
                    break
                constrained += 1
        
        c_xy = self.perturb(cur_perturbation, vel)

        if scale <= 0: 
            c_xy = self.xy

        return c_xy

class DynamicStats(object):
    def __init__(self):
        self.stats = {
            'k': 0, # curvature 
            'dk': 0,
            'ds': 0,
            'a_s': 0, # acc
            'a_l': 0, # lateral acc
            'dh': 0,
            'ddh': 0,
        }
        for key in self.stats.keys():
            self.stats[key] = [np.inf,-np.inf]

        self.dt = 0.5

    def update_stats(self, states):
        for key in states.keys():
            self.stats[key][0] = np.min([states[key].min(),self.stats[key][0]])
            self.stats[key][1] = np.max([states[key].max(),self.stats[key][1]])
        print(self.stats)

    def parse_dynamics(self, pre_motion):
        def d(x):
            dx = x[1:] - x[:-1]
            return dx/self.dt # scaling t
            
        for xy in pre_motion:

            dx = d(xy[:,0])
            dy = d(xy[:,1])
            
            ddx = d(dx)
            ddy = d(dy)

            h = np.arctan2(dy,dx)
            dh = h[1:] - h[:-1]
            for dh_it in range(len(dh)):
                if dh[dh_it] < -np.pi:
                    dh[dh_it] += np.pi * 2
                elif dh[dh_it] > np.pi:
                    dh[dh_it] -= np.pi * 2
            dh = dh / self.dt
            ddh = d(dh)

            # mask_h = (dx * dx + dy * dy == 0) # static
            # dh[mask_h] = 0
            # ddh[mask_h] = 0


            ds = np.sqrt(dx**2 + dy**2)
            dds = d(ds)

            # mask_k = (dx * dx + dy * dy <= 5**2)
            # k = (np.abs(ddx * dy - dx * ddy) / (dx * dx + dy * dy)**1.5)
            # k[mask_k] = 0 # curvature = 0 if speed < 5
            # dk = d(k)

            k = dh/ds[:-1]
            dk = d(k)

            a_l = k * (ds[:-1] ** 2)

            states = {
                'k': k, # curvature 
                'dk': dk,
                'ds': ds,
                'a_s': dds, # acc
                'a_l': a_l, # lateral acc
                'dh': dh,
                'ddh': ddh,
            }
            print(states)
            for value in states.values():
                if np.nan in value:
                    st()
            self.update_stats(states)