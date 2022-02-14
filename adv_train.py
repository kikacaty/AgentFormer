import os
import sys
import argparse
import time
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from model.model_lib import model_dict
from utils.torch import *
from utils.config import Config
from utils.utils import prepare_seed, print_log, AverageMeter, convert_secs2time, get_timestring

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

from pdb import set_trace as st

def motion_bending_loss(fut_motion, pred_motion):
    try:
        diff = (fut_motion - pred_motion).cpu()
    except:
        st()
    loss_unweighted = diff.pow(2).sum(2)
    bend_weight = torch.from_numpy(np.array([(12-i)**3 for i in range(12)]))
    loss_weighted = (diff.pow(2).sum(2) * bend_weight).sum()
    return loss_weighted

def logging(cfg, epoch, total_epoch, iter, total_iter, ep, seq, frame, losses_str, log):
	print_log('{} | Epo: {:02d}/{:02d}, '
		'It: {:04d}/{:04d}, '
		'EP: {:s}, ETA: {:s}, seq {:s}, frame {:05d}, {}'
        .format(cfg, epoch, total_epoch, iter, total_iter, \
		convert_secs2time(ep), convert_secs2time(ep / iter * (total_iter * (total_epoch - epoch) - iter)), seq, frame, losses_str), log)


def gen_adv_data(data, alpha = 1e-2, eps = 1, iterations=5):
    in_data = data

    if len(data['pre_motion_3D']) <= 1:
        return

    ori_pre_motion = model.data['pre_vel']
    perturb_mask = torch.zeros_like(model.data['pre_vel'])
    perturb_mask[...,0] = 1
    target_id = 1

    target_fut_motion = model.data['fut_motion'][:,1:,:].transpose(0,1)
    for i in range(iterations):
        perturbed_pre_motion = model.data['pre_vel'].data
        # for key in model.data.keys():
        #     if model.data[key]
        #     model.data[key].requires_grad = False
        model.data['pre_vel'].requires_grad = True

        recon_motion_3D, _ = model.inference(mode='recon')
        # sample_motion_3D, data = model.inference(mode='infer', sample_num=1, need_weights=False)
        # sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()

        # recon_motion_3D = sample_motion_3D[0]

        target_pred_motion = recon_motion_3D[1:]
        model.zero_grad()

        cost = motion_bending_loss(target_fut_motion, target_pred_motion).to(device)
        # print(cost.item())
        # cost.backward(retain_graph=True)
        cost.backward()

        adv_pre_motion = perturbed_pre_motion + alpha*perturb_mask*model.data['pre_vel'].grad.sign()
        eta = torch.clamp(adv_pre_motion - ori_pre_motion, min=-eps, max=eps)
        model.data['pre_vel'] = (ori_pre_motion + eta).detach_()
        
    
        update_pre_motion = torch.zeros_like(model.data['pre_motion'])
        update_pre_motion[1:,...] = torch.cumsum(model.data['pre_vel'],dim=0) 
        update_pre_motion += model.data['pre_motion'][0,...]
        model.update_data_train(update_pre_motion, in_data)

def train(epoch, args):
    global tb_ind
    since_train = time.time()
    generator.shuffle()
    train_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}
    train_loss_meter['total_loss'] = AverageMeter()
    last_generator_index = 0
    while not generator.is_epoch_end():
        data = generator()
        if data is not None:
            seq, frame = data['seq'], data['frame']

            model.set_data(data)

            gen_adv = True
            if args.mix:
                gen_adv = np.random.random() > 0.5

            if gen_adv:
                model.eval()
                gen_adv_data(data)

            model.train()
            model_data = model()
            total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()
            """ optimize """

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss_meter['total_loss'].update(total_loss.item())
            for key in loss_unweighted_dict.keys():
                train_loss_meter[key].update(loss_unweighted_dict[key])

        if generator.index - last_generator_index > cfg.print_freq:
            ep = time.time() - since_train
            losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
            logging(args.cfg, epoch, cfg.num_epochs, generator.index, generator.num_total_samples, ep, seq, frame, losses_str, log)
            for name, meter in train_loss_meter.items():
                tb_logger.add_scalar('adv_model_' + name, meter.avg, tb_ind)
            tb_ind += 1
            last_generator_index = generator.index

    scheduler.step()
    model.step_annealer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='k10_res')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mix', action='store_true', default=False)

    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg, args.tmp, create_dirs=True)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    
    time_str = get_timestring()
    log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')
    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)
    tb_logger = SummaryWriter(cfg.tb_dir)
    tb_ind = 0

    """ data """
    generator = data_generator(cfg, log, split='train', phase='training')

    """ model """
    model_id = cfg.get('model_id', 'agentformer')
    model = model_dict[model_id](cfg)
    model.set_device(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler_type = cfg.get('lr_scheduler', 'linear')
    if scheduler_type == 'linear':
        scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
    elif scheduler_type == 'step':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')

    if args.start_epoch > 0:
        cp_path = cfg.model_path % args.start_epoch
        print_log(f'loading model from checkpoint: {cp_path}', log)
        model_cp = torch.load(cp_path, map_location='cpu')
        # model_cp = torch.load(cp_path, map_location=device)
        model.load_state_dict(model_cp['model_dict'])
        if 'opt_dict' in model_cp:
            optimizer.load_state_dict(model_cp['opt_dict'])
        if 'scheduler_dict' in model_cp:
            scheduler.load_state_dict(model_cp['scheduler_dict'])

    """ start training """
    model.set_device(device)
    model.train()

    for i in range(args.start_epoch, cfg.num_epochs):
        train(i,args)
        """ save model """
        if cfg.model_save_freq > 0 and (i + 1) % cfg.model_save_freq == 0:
            cp_path = cfg.model_path % (i + 1)
            model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(), 'scheduler_dict': scheduler.state_dict(), 'epoch': i + 1}
            torch.save(model_cp, cp_path)

