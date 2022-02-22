import sys, os
sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.config import Config, AdvConfig
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing
from data.nuscenes_pred_split import get_nuscenes_pred_split

import numpy as np

from tqdm import tqdm

from pdb import set_trace as st

from utils.attack_utils.constraint import DynamicModel, DynamicStats

def main():
    data_root = 'datasets/nuscenes_pred'
    seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
    split = 'train'

    d_stats = DynamicStats()

    for seq_name in seq_train:
        print(f'parsing seq {seq_name}...')
        label_path = os.path.join(data_root, 'label/{}/{}.txt'.format(split, seq_name))
        delimiter = ' '

        gt = np.genfromtxt(label_path, delimiter=delimiter, dtype=str)
        gt[:,0] = frames = gt[:,0].astype(np.float).astype(np.int)
        gt[:,1] = ids = gt[:,1].astype(np.float).astype(np.int)
        gt[:,[13,15]] = xys = gt[:,[13,15]].astype(np.float)
        
        u_ids = np.unique(ids)

        for cur_id in u_ids:
            traj = []
            for idx, id in enumerate(ids):
                if id == cur_id: 
                    traj.append(xys[idx,:])
            traj = np.array(traj)
            d_stats.parse_dynamics(traj)

    print('name\t mean\t var\t num\t min\t max')
    for key in d_stats.stats.keys():
        print(f'{key}\t'+'\t'.join([f'{item:.3f}' for item in d_stats.stats[key]]))

if __name__ == '__main__':
    main()