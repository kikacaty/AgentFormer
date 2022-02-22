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

from tqdm import tqdm

from matplotlib import collections  as mc
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  
from matplotlib.lines import Line2D
from pdb import set_trace as st

from utils.attack_utils.constraint import DynamicModel, DynamicStats
from utils.attack_utils.attack import Attacker

from attack import attack

import torch.multiprocessing as mp

import pickle, yaml

import re

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


    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
    adv_cfg = AdvConfig(args.adv_cfg)
    adv_cfg.sample_k = cfg.sample_k
    adv_cfg.traj_scale = cfg.traj_scale

    p_list = []

    if adv_cfg.mode == 'opt':
        step_size_dds_list = [0.5,0.1,0.05,0.01]
        step_size_dk_list = [0.5,0.1,0.05,0.01]
        for step_size_dds in step_size_dds_list:
            p_list = []

            for step_size_dk in step_size_dk_list:
                for fix_t in [0,-1]:
                    adv_cfg.step_size_dds = step_size_dds
                    adv_cfg.step_size_dk = step_size_dk
                    adv_cfg.fix_t.t_idx = fix_t
                    fix_name = 'end' if adv_cfg.fix_t.t_idx == -1 else 'start'
                    exp_name = f'sweeps/{adv_cfg.mode}_{fix_name}_dds_{adv_cfg.step_size_dds}_dk_{adv_cfg.step_size_dk}'

                    adv_cfg.exp_name = exp_name

                    cfg.adv_cfg = adv_cfg

                    p = mp.Process(target=attack, args=(args, cfg, adv_cfg,))
                    p.start()
                    p_list.append(p)
                    # attack(args, cfg, adv_cfg)

            for p in p_list:
                p.join()

    else:
        step_size_list = [0.5,0.1,0.05,0.01]
        for step_size in step_size_list:
            exp_name = f'sweeps/{adv_cfg.mode}_step_size_{adv_cfg.step_size}'
            adv_cfg.step_size = step_size
            adv_cfg.exp_name = exp_name

            cfg.adv_cfg = adv_cfg

            p = mp.Process(target=attack, args=(args, cfg, adv_cfg,))
            p.start()
            p_list.append(p)
            # attack(args, cfg, adv_cfg)

    for p in p_list:
        p.join()