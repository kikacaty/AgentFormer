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
from utils.config import Config, AdvConfig
from utils.utils import prepare_seed, print_log, AverageMeter, convert_secs2time, get_timestring

from utils.attack_utils.attack import Attacker, trade_loss, simple_noise_attack

from utils.timer import Timer
from tqdm import tqdm

import wandb

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

torch.set_printoptions(sci_mode=False, precision=3)


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
    print('{} | Epo: {:02d}/{:02d}, '
		'It: {:04d}/{:04d}, '
		'EP: {:s}, ETA: {:s}, seq {:s}, frame {:05d}, {}'
        .format(cfg, epoch, total_epoch, iter, total_iter, \
		convert_secs2time(ep), convert_secs2time(ep / iter * (total_iter * (total_epoch - epoch) - iter)), seq, frame, losses_str))
	
    # print_log('{} | Epo: {:02d}/{:02d}, '
	# 	'It: {:04d}/{:04d}, '
	# 	'EP: {:s}, ETA: {:s}, seq {:s}, frame {:05d}, {}'
    #     .format(cfg, epoch, total_epoch, iter, total_iter, \
	# 	convert_secs2time(ep), convert_secs2time(ep / iter * (total_iter * (total_epoch - epoch) - iter)), seq, frame, losses_str), log)


def validate(epoch, args):
    train_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}

    adv_train_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}

    eval_scenes = set()
    model.eval()

    pbar = tqdm(total = test_generator.num_total_samples)

    while not test_generator.is_epoch_end():
        data = test_generator()
        pbar.update(1)
        if data is not None:
            seq, frame = data['seq'], data['frame']
            if seq in eval_scenes: continue
            eval_scenes.add(seq)

            model.eval()
            model.set_data(data)
            model()
            total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()

            for key in loss_unweighted_dict.keys():
                train_loss_meter[key].update(loss_unweighted_dict[key])

            simple_noise_attack(model,data,eps=args.eps/10, iters=args.test_pgd_step,qz=args.qz)
            if args.fixed:
                model.adv_forward()
            else:
                model()
            total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()

            for key in loss_unweighted_dict.keys():
                adv_train_loss_meter[key].update(loss_unweighted_dict[key])

            pbar.set_description(' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in adv_train_loss_meter.items()]))

    # losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in adv_train_loss_meter.items()])
    # print(f'Validation for Epo {epoch}: {losses_str}')

    if not args.debug:
        wandb_log = {}
        for x, y in adv_train_loss_meter.items():
            if x == 'kld' or x == 'total_loss': continue
            wandb_log[f'test/{x}'] = y.avg
        for x, y in train_loss_meter.items():
            if x == 'kld' or x == 'total_loss': continue
            wandb_log[f'test/benign_{x}'] = y.avg
        wandb.log(wandb_log)

def train(epoch, args):
    global tb_ind
    since_train = time.time()
    generator.shuffle()
    train_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}
    train_loss_meter['total_loss'] = AverageMeter()
    if args.qz_reg:
        train_loss_meter['qz'] = AverageMeter()
    if args.context_reg:
        train_loss_meter['context'] = AverageMeter()
    last_generator_index = 0
    # attacker = Attacker(model, adv_cfg)

    adv_timer = Timer()
    train_timer = Timer()

    while not generator.is_epoch_end():

        if args.dense and generator.index % cfg.validate_freq == 0:
            validate(epoch, args)
            
        data = generator()

        if data is not None:
            seq, frame = data['seq'], data['frame']

            if args.benign:
                model.train()
                model.set_data(data)
                model_data = model()
                total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()
            else:
                adv_timer.tic()
                if args.trade:
                    model.eval()
                    loss_trade = trade_loss(model, data, eps=args.eps/10, iters=args.pgd_step)

                    model.train()
                    model.set_data(data)
                    model_data = model()
                    total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()

                    total_loss += args.beta * loss_trade
                else:
                    if args.all:
                        model.train()
                        model.set_data(data)
                        model_data = model()
                        benign_total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()

                    model.eval()
                    adv_data_out = simple_noise_attack(model, data, eps=args.eps/10, iters=args.pgd_step, qz=args.qz, context=args.context, naive=args.naive)
                    model.train()
                    model_data = model()
                    total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()
                    if args.qz_reg:
                        qz_loss = model.compute_qz_loss()
                        total_loss += args.qz_reg_beta * qz_loss
                        loss_unweighted_dict['qz'] = qz_loss.item()
                    if args.context_reg:
                        ctx_loss = model.compute_ctx_loss()
                        total_loss += args.context_reg_beta * ctx_loss
                        loss_unweighted_dict['context'] = ctx_loss.item()
                    if args.all:
                        total_loss += benign_total_loss
                    
                if args.debug:
                    print(f'adv time: {adv_timer.toc()}')

            train_timer.tic()
            """ optimize """
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if args.debug:
                print(f'train time: {train_timer.toc()}')

            train_loss_meter['total_loss'].update(total_loss.item())
            for key in loss_unweighted_dict.keys():
                train_loss_meter[key].update(loss_unweighted_dict[key])

        if not args.debug and generator.index - last_generator_index > cfg.print_freq:
            ep = time.time() - since_train
            losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
            logging(args.cfg, epoch, cfg.num_epochs, generator.index, generator.num_total_samples, ep, seq, frame, losses_str, log)
            last_generator_index = generator.index

            wandb_log = {}
            for x, y in train_loss_meter.items():
                wandb_log[x] = y.avg
            wandb_log['epoch'] = epoch
            wandb.log(wandb_log)

    scheduler.step()
    model.step_annealer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='k10_res')
    parser.add_argument('--adv_cfg', default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ngc', action='store_true', default=False)

    parser.add_argument('--free', action='store_true', default=False)
    parser.add_argument('--full', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--dense', action='store_true', default=False)

    # adv train params
    parser.add_argument('--beta', type=float, default=5)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--pgd_step', type=int, default=1)
    parser.add_argument('--test_pgd_step', type=int, default=10)

    # parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--trade', action='store_true', default=False)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--finetune_lr', type=float, default=0.1)
    parser.add_argument('--finetune_fast', action='store_true', default=False)

    parser.add_argument('--benign', action='store_true', default=False)
    parser.add_argument('--fixed', action='store_true', default=False)
    parser.add_argument('--qz', action='store_true', default=False)

    parser.add_argument('--context', action='store_true', default=False)

    parser.add_argument('--qz_reg', action='store_true', default=False)
    parser.add_argument('--qz_reg_beta', type=float, default=1)

    parser.add_argument('--context_reg', action='store_true', default=False)
    parser.add_argument('--context_reg_beta', type=float, default=1)

    parser.add_argument('--naive', action='store_true', default=False)

    parser.add_argument('--all', action='store_true', default=False)





    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg, tmp=args.tmp, create_dirs=True, ngc=args.ngc)
    adv_cfg = AdvConfig(args.adv_cfg, tmp=args.tmp, create_dirs=True)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    
    
    adv_cfg.device = device
    adv_cfg.traj_scale = cfg.traj_scale

    if args.full:
        adv_cfg.target_agent.all = True
        adv_cfg.target_agent.other = False
        adv_cfg.adv_agent.all = True
    else:
        adv_cfg.target_agent.all = True
        adv_cfg.target_agent.other = True
        adv_cfg.adv_agent.all = False

    adv_cfg.iters = [args.pgd_step]

    args.finetune = (args.pretrained is not None)

    adv_agent = 'full' if args.full else 'single'
    exp_name = f'eps_{args.eps}_step_{args.pgd_step}_free_{args.free}_fixed_{args.fixed}_qz_{args.qz}_ctx_{args.context}_adv'
    if args.trade:
        exp_name = f'trade_{args.beta}/{exp_name}'
    if args.qz_reg:
        exp_name = f'qz_reg_{args.qz_reg_beta}/{exp_name}'
    if args.context_reg:
        exp_name = f'ctx_reg_{args.context_reg_beta}/{exp_name}'

    if args.naive:
        exp_name = f'eps_{args.eps}_step_{args.pgd_step}_naive_adv'
    if args.finetune:
        cfg.lr *= args.finetune_lr
        if args.finetune_fast:
            cfg.decay_step = int(cfg.decay_step/10)
            exp_name = f'fast_finetune_{args.finetune_lr}/{exp_name}'
        else:
            exp_name = f'finetune_{args.finetune_lr}/{exp_name}'

    if args.all:
        exp_name = f'all/{exp_name}'

    # linf attack
    exp_name = f'linf/{exp_name}'

    if not args.ngc:
        exp_name = f'fast/{exp_name}'

    # exp_name = f'pgd_step_{args.pgd_step}_mix_{args.mix}_free_{args.free}_adv_{adv_cfg.mode}_{adv_agent}'

    if not args.debug:
        # set up wandb
        wandb.init(project="robust_pred", entity="yulongc")
        wandb.config = {
            'pgd_step': args.pgd_step,
            'mix': args.mix,
            'free': args.free,
            'adv mode': adv_cfg.mode,
            'beta': args.beta,
            'eps': args.eps
        }

        
        wandb.run.name = exp_name
        wandb.run.save()
    cfg.update_dirs(exp_name)
    
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
    test_generator = data_generator(cfg, log, split='test', phase='testing')

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

    if args.pretrained:
        cp_path = args.pretrained
        print_log(f'loading model from checkpoint: {cp_path}', log)
        # model_cp = torch.load(cp_path, map_location='cpu')
        model_cp = torch.load(cp_path, map_location=device)
        model.load_state_dict(model_cp['model_dict'],strict=False)
        # if 'opt_dict' in model_cp:
        #     optimizer.load_state_dict(model_cp['opt_dict'])
        # if 'scheduler_dict' in model_cp:
        #     scheduler.load_state_dict(model_cp['scheduler_dict'])


    """ start training """
    model.set_device(device)
    model.train()

    if not args.debug:
        validate(0,args)

    for i in range(args.start_epoch, cfg.num_epochs):
        train(i,args)

        validate(i+1,args)
        """ save model """
        if cfg.model_save_freq > 0 and (i + 1) % cfg.model_save_freq == 0:
            cp_path = cfg.model_path % (i + 1)
            model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(), 'scheduler_dict': scheduler.state_dict(), 'epoch': i + 1}
            torch.save(model_cp, cp_path)

