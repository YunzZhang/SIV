# -*- coding: utf-8 -*-
import argparse
import os
import time
import cv2
import os.path as osp
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import pprint
from model.utils import *
from logger import *
from configs.yml_parser import YAMLParser
from easydict import EasyDict

from model.get_model import get_model
from datasets.dat_loader import *
import tqdm
import warnings
from tqdm import trange
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--group', '-g', type=str, default='.')
parser.add_argument('--configs', '-c', type=str, default='./configs/Problem1.yml')
parser.add_argument('--save_dir', '-sd', type=str, default='./outputs')
parser.add_argument('--batch_size', '-bs', type=int, default=4)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--num_workers', '-j', type=int, default=12)
parser.add_argument('--start_epoch', '-se', type=int, default=0)
parser.add_argument('--pretrained', '-prt', type=str, default=None) 
parser.add_argument('--print_freq', '-pf', type=int, default=None)
parser.add_argument('--vis_path', '-vp', type=str, default='./vis')
parser.add_argument('--model_iters', '-mit', type=int, default=8)
parser.add_argument('--no_warm', '-nw', action='store_true', default=False)
parser.add_argument('--eval', '-e', action='store_true', default=False)
parser.add_argument('--save_name', '-sn', type=str, default=None)
parser.add_argument('--warm_iters', '-wi', type=int, default=3000)
parser.add_argument('--eval_vis', '-ev', type=str, default='eval_vis')
parser.add_argument('--crop_len', '-clen', type=int, default=400)
parser.add_argument('--with_valid', '-wv', type=bool, default=True)
parser.add_argument('--decay_interval', '-di', type=int, default=10)
parser.add_argument('--decay_factor', '-df', type=float, default=0.85)
parser.add_argument('--valid_vis_freq', '-vvf', type=int, default=40)
parser.add_argument('--valid_freq', '-vf', type=int, default=10)
parser.add_argument('--dt', '-dt', type=int, default=21)
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
# parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
args = parser.parse_args()

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
cfg_parser = YAMLParser(args.configs)
cfg = cfg_parser.config

n_iter = 0

args.save_dir = os.path.join(args.group,args.save_dir)
args.eval_vis = os.path.join(args.group,args.eval_vis)
args.vis_path = os.path.join(args.group,args.vis_path)

if args.print_freq != None:
    cfg['train']['print_freq'] = args.print_freq
if args.batch_size != None:
    cfg['loader']['batch_size'] = args.batch_size


################################# set dt ###################################
cfg['data']['dt'] = args.dt

################################# warm up ###################################
warmup = WarmUp(ed_it=args.warm_iters, st_lr=1e-7, ed_lr=args.learning_rate)
#############################################################################

## Train
def train(cfg, train_loader, model, optimizer, epoch, log, train_writer):
    ######################################################################
    ## Init
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()

    ######################################################################
    ## Training Loop
    
    for ww, data in enumerate(train_loader, 0):
        if (not args.no_warm) and (n_iter <= args.warm_iters):
            warmup.adjust_lr(optimizer=optimizer, cur_it=n_iter)
        
        # spike = data['spike']
        spks = data['spike']
        spks = spks[:,10:52,:,:]
        if cfg['data']['dt'] == 21:
            flowgt = data['flows'][0].cuda() # dt=21
        elif cfg['data']['dt'] == 11:
            flowgt = data['flows'][1].cuda() # dt=11
        else:
            raise ValueError("dt should be 21 or 11")
        data_time.update(time.time() - end)

        if args.model_iters == None:
            flow_pred = model(spks=spks,dt=args.dt)
        else:
            flow_pred = model(spks=spks, iters=args.model_iters,dt=args.dt)
        

        ## compute loss
        loss, loss_deriv_dict = compute_loss(flow_pred, flowgt)
        
        # record loss
        losses.update(loss.item())
        flow_mean = loss_deriv_dict['flow_mean']
        train_writer.add_scalar('total_loss', loss.item(), n_iter)
        train_writer.add_scalar('flow_mean', flow_mean, n_iter)

        ## compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        n_iter += 1
        if n_iter % cfg['train']['vis_freq'] == 0:
            vis_flow(flow_pred[-1], args.vis_path, suffix='forw_flow_{:02d}'.format(int(args.dt)))
            vis_flow(flowgt, args.vis_path, suffix='GT_flow_{:02d}'.format(int(args.dt)))

        # output logs
        if ww % cfg['train']['print_freq'] == 0:
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            out_str = 'Epoch: [{:d}] [{:d}/{:d}],  Iter: {:d}  '.format(epoch, ww, len(train_loader), n_iter-1)
            out_str += 'Time: {},  Data: {},  Loss: {}, Flow mean {:.4f}, lr {:.7f}'.format(batch_time, data_time, losses, flow_mean, cur_lr)
            log.info(out_str)

        end = time.time()
    
    return


##########################################################################################################
## valid
def validation(cfg, test_datasets, model, log):
    global n_iter
    data_time = AverageMeter()
    AEE = AverageMeter()
    RMSE = AverageMeter()
    model_time = AverageMeter()
    end = time.time()
    epe_dict = {}
    rmse_dict = {}

    len_dict = {}

    # switch to evaluate mode
    model.eval()

    i_set = 0
    for sub_dataset, cur_test_set in test_datasets.items():
        i_set += 1
        cur_test_loader = torch.utils.data.DataLoader(
            cur_test_set,
            pin_memory = False,
            drop_last = False,
            batch_size = 1,
            shuffle = False,
            num_workers = args.num_workers)

        cur_aee = AverageMeter()
        cur_rmse = AverageMeter()

        cur_model_time = AverageMeter()
        cur_eval_vis_path = osp.join(args.eval_vis , "{:02d}".format(int(args.dt)), sub_dataset)
        make_dir(cur_eval_vis_path)
        for ww, data in enumerate(cur_test_loader, 0):

            spks = data['spike']
            spks = spks[:,10:52,:,:]
            if cfg['data']['dt'] == 21:
                flowgt = data['flows'][0].cuda().permute([0,3,1,2]) # dt=21
            elif cfg['data']['dt'] == 11:
                flowgt = data['flows'][1].cuda().permute([0,3,1,2]) # dt=11
            else:
                raise ValueError("dt should be 21 or 11")

            data_time.update(time.time() - end)
            with torch.no_grad():
                st = time.time()
                flow = model(spks=spks, iters=args.model_iters,dt=args.dt)
                mtime = time.time() - st

            if ww % args.valid_vis_freq == 0:
                flow_vis = flow_to_image(flow[-1][0].permute([1,2,0]).cpu().numpy())
                cur_vis_path = osp.join(cur_eval_vis_path, '{:03d}.png'.format(ww))
                cv2.imwrite(cur_vis_path, flow_vis)

            # epe
            epe = torch.norm(flow[-1] - flowgt, p=2, dim=1).mean()

            rmse = calculate_rmse(flow[-1], flowgt)

            cur_aee.update(epe)
            cur_rmse.update(rmse)

            AEE.update(epe)
            RMSE.update(rmse)

            cur_model_time.update(mtime)
            model_time.update(mtime)
        epe_dict[sub_dataset] = cur_aee.avg
        rmse_dict[sub_dataset] = cur_rmse.avg

        len_dict[sub_dataset] = cur_test_set.__len__()
        log.info('sub-Dataset[{:02d}]: {:30s}  EPE: {:.4f}  RMSE: {:.4f}  AvgTime: {:.4f}'.format(
            i_set, sub_dataset, cur_aee.avg, cur_rmse.avg, cur_model_time.avg))
        time.sleep(0.1)
    
    log.info('All EPE/RMSE: {:.4f}/{:.4f}  AvgTime: {:.4f}'.format(
        AEE.avg, RMSE.avg, model_time.avg))
    a_epe, b_epe, c_epe, d_epe = get_class_metric(epe_dict, len_dict)

    a_rmse, b_rmse, c_rmse, d_rmse = get_class_metric(rmse_dict, len_dict)


    log.info('EPE/RMSE: Class A: {:.4f}, {:.4f}  Class B: {:.4f}, {:.4f}  Class C: {:.4f}, {:.4f}  Class D: {:.4f}, {:.4f}'.format(a_epe, a_rmse, b_epe, b_rmse, c_epe, c_rmse, d_epe, d_rmse))

    
    return AEE.avg



if __name__ == '__main__':
    ##########################################################################################################
    # Create save path and logs
    timestamp1 = datetime.datetime.now().strftime('%m-%d')
    timestamp2 = datetime.datetime.now().strftime('%H%M%S')
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    if args.save_name == None:
        save_folder_name = 'b{:d}_{:s}'.format(cfg['loader']['batch_size'], timestamp2)
    else:
        save_folder_name = 'b{:d}_{:s}_{:s}'.format(cfg['loader']['batch_size'], timestamp2, args.save_name)

    save_root = osp.join(args.save_dir, "Problem{}".format(int(cfg['data']['Problem_type'])), "dt={:02d}".format(int(args.dt)), timestamp1)
    save_path = osp.join(save_root, save_folder_name)
    make_dir(args.save_dir)
    make_dir(save_root)
    make_dir(save_path)
    args.vis_path = os.path.join(args.vis_path, "Problem{}".format(int(cfg['data']['Problem_type'])), "dt={:02d}".format(int(args.dt)))
    make_dir(args.vis_path)
    args.eval_vis = os.path.join(args.eval_vis, "Problem{}".format(int(cfg['data']['Problem_type'])), "dt={:02d}".format(int(args.dt)))
    make_dir(args.eval_vis)

    _log = init_logger(log_dir=save_path, filename=timestamp2+'.log')
    _log.info('=> will save everything to {:s}'.format(save_path))
    # show configurations
    cfg_str = pprint.pformat(cfg)
    _log.info('=> configurations: \n' + cfg_str)

    train_writer = SummaryWriter(save_path)

    ##########################################################################################################
    ## Create model
    model_dict =  {
            "dropout": 0.0,
            "mixed_precision": args.mixed_precision,
            }
    model_dict = EasyDict(model_dict)
    model = get_model(model_dict)
    
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        _log.info('=> using pretrained flow model {:s}'.format(args.pretrained))
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(network_data)
    else:
        network_data = None
        _log.info('=> train flow model from scratch')
        model.init_weights()
        _log.info('=> Flow model params: {:.6f}M'.format(model.num_parameters()/1e6))
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    ##########################################################################################################
    ## Create Optimizer
    cfgopt = cfg['optimizer']
    cfgmdl = cfg['model']
    assert(cfgopt['solver'] in ['Adam', 'SGD'])
    _log.info('=> settings {:s} solver'.format(cfgopt['solver']))
    
    param_groups = [{'params': model.module.parameters(), 'weight_decay': cfgmdl['flow_weight_decay']}]
    if cfgopt['solver'] == 'Adam':
        optimizer = torch.optim.Adam(param_groups, args.learning_rate, betas=(cfgopt['momentum'], cfgopt['beta']))
    elif cfgopt['solver'] == 'SGD':
        optimizer = torch.optim.SGD(param_groups, args.learning_rate, momentum=cfgopt['momentum'])
        
    ##########################################################################################################
    # Dataset
    train_set = SpkLoader_train(cfg)
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        drop_last = True,
        batch_size = cfg['loader']['batch_size'],
        shuffle = True,
        num_workers = args.num_workers)
    
    if args.eval:
        test_datasets = get_test_datasets(cfg, valid=False)
        validation(cfg=cfg, test_datasets=test_datasets, model=model, log=_log)
    else: # validation during training
        test_datasets = get_test_datasets(cfg, valid=True, crop_len=args.crop_len)
        epoch = args.start_epoch
        n_epochs = cfg['loader']['n_epochs']
        for iter in trange(epoch, n_epochs):
            train(
                cfg=cfg,
                train_loader=train_loader,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                log=_log,
                train_writer=train_writer)
            epoch += 1

            if epoch % args.decay_interval == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * args.decay_factor

            if args.with_valid:
                best_AEE = 100
                if epoch % args.valid_freq == 0:
                    AEE = validation(
                            cfg=cfg, 
                            test_datasets=test_datasets,
                            model=model, 
                            log=_log)
                    # if AEE < best_AEE:
                    #     best_model_save_name = '{:s}_best.pth'.format(cfg['model']['flow_arch'])

                    # Save Model
                    flow_model_save_name = '{:s}_epoch{:03d}.pth'.format(cfg['model']['flow_arch'], epoch)
                    torch.save(model.state_dict(), osp.join(save_path, flow_model_save_name))


            if epoch >= cfg['loader']['n_epochs']:
                break