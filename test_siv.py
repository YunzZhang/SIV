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
import sys
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
from vis_epe import *

parser = argparse.ArgumentParser()
parser.add_argument('--group', '-g', type=str, default='.')
parser.add_argument('--configs', '-c', type=str, default='./configs/Problem1.yml')
# parser.add_argument('--save_dir', '-sd', type=str, default='./outputs')
parser.add_argument('--batch_size', '-bs', type=int, default=4)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--num_workers', '-j', type=int, default=12)
parser.add_argument('--start_epoch', '-se', type=int, default=0)
parser.add_argument('--pretrained', '-prt', type=str, default='pretrained/dt=21/siv_p1_epoch100.pth') 
parser.add_argument('--print_freq', '-pf', type=int, default=None)
# parser.add_argument('--vis_path', '-vp', type=str, default='./vis')
parser.add_argument('--model_iters', '-mit', type=int, default=8)
parser.add_argument('--no_warm', '-nw', action='store_true', default=False)
parser.add_argument('--eval', '-e', action='store_true', default=True)
parser.add_argument('--save_name', '-sn', type=str, default=None)
parser.add_argument('--warm_iters', '-wi', type=int, default=3000)
parser.add_argument('--eval_vis', '-ev', type=str, default='./eval_vis')
parser.add_argument('--crop_len', '-clen', type=int, default=400)
parser.add_argument('--with_valid', '-wv', type=bool, default=True)
parser.add_argument('--decay_interval', '-di', type=int, default=10)
parser.add_argument('--decay_factor', '-df', type=float, default=0.85)
parser.add_argument('--valid_vis_freq', '-vvf', type=int, default=10)
parser.add_argument('--valid_freq', '-vf', type=int, default=10)
parser.add_argument('--dt', '-dt', type=int, default=21)
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
# parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
args = parser.parse_args()

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
cfg_parser = YAMLParser(args.configs)
cfg = cfg_parser.config

n_iter = 0


if args.print_freq != None:
    cfg['train']['print_freq'] = args.print_freq
if args.batch_size != None:
    cfg['loader']['batch_size'] = args.batch_size


################################# set dt ###################################
cfg['data']['dt'] = args.dt



##########################################################################################################
## valid
def validation(cfg, test_datasets, model):
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
        cur_epe_vis_path = cur_eval_vis_path.replace("vis","epe_vis")
        make_dir(cur_eval_vis_path)
        make_dir(cur_epe_vis_path)
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
            epe_map = torch.norm(flow[-1] - flowgt, p=2, dim=1).permute([1,2,0]).cpu().detach().numpy()
            if ww % args.valid_vis_freq == 0:
                flow_vis = flow_to_image(flow[-1][0].permute([1,2,0]).cpu().numpy())
                cur_vis_path = osp.join(cur_eval_vis_path, '{:03d}_siv.png'.format(ww))
                epe_vis = visualize_epemap(epe_map,osp.join(cur_epe_vis_path, '{:03d}_siv.png'.format(ww)),cmap='plasma')
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
        print('sub-Dataset[{:02d}]: {:30s}  EPE: {:.4f}  RMSE: {:.4f}  AvgTime: {:.4f}'.format(
            i_set, sub_dataset, cur_aee.avg, cur_rmse.avg, cur_model_time.avg))
        time.sleep(0.1)
    
    print('All EPE/RMSE: {:.4f}/{:.4f}  AvgTime: {:.4f}'.format(
        AEE.avg, RMSE.avg, model_time.avg))
    a_epe, b_epe, c_epe, d_epe = get_class_metric(epe_dict, len_dict)

    a_rmse, b_rmse, c_rmse, d_rmse = get_class_metric(rmse_dict, len_dict)


    print('EPE/RMSE: Class A: {:.4f}, {:.4f}  Class B: {:.4f}, {:.4f}  Class C: {:.4f}, {:.4f}  Class D: {:.4f}, {:.4f}'.format(a_epe, a_rmse, b_epe, b_rmse, c_epe, c_rmse, d_epe, d_rmse))

    
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


    args.eval_vis = os.path.join(args.eval_vis, "Problem{}".format(int(cfg['data']['Problem_type'])), "dt={:02d}".format(int(args.dt)))
    make_dir(args.eval_vis)

    # show configurations
    cfg_str = pprint.pformat(cfg)

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
        print('=> using pretrained flow model {:s}'.format(args.pretrained))
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(network_data)
    else:
        network_data = None
        print('=> train flow model from scratch')
        model.init_weights()
        print('=> Flow model params: {:.6f}M'.format(model.num_parameters()/1e6))
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    ##########################################################################################################
    ## Create Optimizer
    cfgopt = cfg['optimizer']
    cfgmdl = cfg['model']
    assert(cfgopt['solver'] in ['Adam', 'SGD'])
    print('=> settings {:s} solver'.format(cfgopt['solver']))
    
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

    test_datasets = get_test_datasets(cfg, valid=False)
    validation(cfg=cfg, test_datasets=test_datasets, model=model)