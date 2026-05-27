import os
# import h5py
import numpy as np
import os.path as osp
import torch
import random
from datasets.ds_utils import *

TEST_SUBDATASET = [
    'cha',
    'iso',
    'mhd',
    'mix'
]

class Augmentor:
    def __init__(self, do_flip):
        # spatial augmentation params
        self.spatial_aug_prob = 0.8

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.2

    def spatial_transform(self, spk, flow_list=None):

        do_lr_flip = np.random.rand() < self.h_flip_prob
        do_ud_flip = np.random.rand() < self.v_flip_prob


        if self.do_flip:
            if do_lr_flip:
                spk = np.flip(spk, axis=2)
            if do_ud_flip:
                spk = np.flip(spk, axis=1)
            
        if flow_list != None:
            for ii, flow in enumerate(flow_list):
                flow = flow.transpose([2, 0, 1])
                if self.do_flip:
                    if do_lr_flip:
                        flow = np.flip(flow, axis=2)
                    if do_ud_flip:
                        flow = np.flip(flow, axis=1)
                        flow[1,:,:] = -flow[1,:,:]
                flow_list[ii] = flow
        
        return spk, flow_list

    def __call__(self, spk, flow_list=None):
        spk, flow_list = self.spatial_transform(spk, flow_list)
        spk = np.ascontiguousarray(spk)
        flow_list = [np.ascontiguousarray(flow) for flow in flow_list]
        return spk, flow_list


class SpkLoader_train(torch.utils.data.Dataset):
    """
        self.cfg:                       config data

    """
    def __init__(self, cfg):
        self.cfg = cfg

    
        # Augmentor
        self.augmentor = Augmentor(do_flip=self.cfg['loader']['do_flip'])
        self.problem_type = self.cfg['data']['Problem_type']
        self.samples = self.collect_samples()
        print('dataset Problem{} - Training, samples num: {:d}'.format(self.problem_type, len(self.samples)))

    def confirm_exist(self, path_list_list):
        for pl in path_list_list:
            for p in pl:
                if not osp.exists(p):
                    return 0
        return 1

    def collect_samples(self):
        Problem_directory = osp.join(self.cfg['data']['Problem_path'], 'train')
        spike_directory = osp.join(Problem_directory,'spk')
        scene_list = []
        for l1_folfer in [os.path.join(spike_directory,fn) for fn in sorted(os.listdir(spike_directory))]: # cha iso
            for scene_path in [os.path.join(l1_folfer,sn) for sn in sorted(os.listdir(l1_folfer))]:
                scene_list.append(scene_path) # complete path: "D:\Dataset\PIV_Dataset_for_Spike_Camera\Problem1\train\spk\cha\p1cha05010063050405"
        random.shuffle(scene_list)
        # scene_list = scene_list[:1000] 
        samples = []
        for scene_path in scene_list:
            scene = scene_path.split("/")[-1]
            spike_dir = scene_path
            flow1_path = scene_path.replace("spk","flo1") + '.flo'
            flow2_path = scene_path.replace("spk","flo2") + '.flo'   
            spike_path = osp.join(spike_dir,'{}.dat'.format(scene))

            if(self.confirm_exist(spike_path) and osp.exists(flow1_path), osp.exists(flow2_path)):
                s = {}
                s['spike_path'] = spike_path
                s['flow1_path'] = flow1_path
                s['flow2_path'] = flow2_path
                s['scene'] = scene[:5] #p1iso
                samples.append(s)
        return samples

    def _load_sample(self, s):
        data = {}
        # h5file = h5py.File(s['spike_path'])
        datfile = dat_to_spmat(s['spike_path'],size=[256,256])
        data['spike'] = np.array(datfile).astype(np.float32)
        data['flows'] = [read_gen(s['flow{:d}_path'.format(ii)]).astype(np.float32) for ii in range(1,3)]
        # Augmentation
        data['spike'], data['flows'] = self.augmentor(data['spike'], data['flows'])
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data


class SpkLoader_test(torch.utils.data.Dataset):
    """
        self.cfg:                       config data

    """
    def __init__(self, cfg, ds_type='test', sub_dataset=None, valid=False, crop_len=500):
        self.cfg = cfg
        self.type = ds_type
        self.sub_dataset = sub_dataset
        self.valid = valid
        self.problem_type = self.cfg['data']['Problem_type']
        ####### Settings of the PIVDSC-Eval #######
        self.crop_len = crop_len
        #########################################

        self.samples = self.collect_samples()
        print('|* Problem {} *| Sub_dataset {}, samples num: {:d}'.format(self.sub_dataset, self.sub_dataset, len(self.samples)))

    def confirm_exist(self, path_list_list):
        for pl in path_list_list:
            for p in pl:
                if not osp.exists(p):
                    return 0
        return 1

    def collect_samples(self):
        Problem_directory = osp.join(self.cfg['data']['Problem_path'], 'test')
        spike_directory = osp.join(Problem_directory,'spk')

        sub_dataset_list = sorted(os.listdir(spike_directory))
        samples = []
        scene_list = []
        for sub_dataset in sub_dataset_list:
            if sub_dataset != self.sub_dataset: # cha iso mhd mix
                continue
            sub_dataset_path = osp.join(spike_directory,sub_dataset)
            for scene_path in [os.path.join(sub_dataset_path,sn) for sn in os.listdir(sub_dataset_path)]:
                scene_list.append(scene_path) # complete path: "D:\Dataset\PIV_Dataset_for_Spike_Camera\Problem1\train\spk\cha\p1cha05010063050405"

        # random.shuffle(scene_list)
        if self.valid:
            scene_list = scene_list[:self.crop_len]
        for scene_path in scene_list:
            scene = scene_path.split("/")[-1]
            spike_dir = scene_path
            flow1_path = scene_path.replace("spk","flo1") + '.flo'
            flow2_path = scene_path.replace("spk","flo2") + '.flo'            
            spike_path = osp.join(spike_dir,'{}.dat'.format(scene))

            if(self.confirm_exist(spike_path) and osp.exists(flow1_path), osp.exists(flow2_path)):
                s = {}
                s['spike_path'] = spike_path
                s['flow1_path'] = flow1_path
                s['flow2_path'] = flow2_path
                s['scene'] = scene[:5] #p1iso
                samples.append(s)
        return samples


    def _load_sample(self, s):
        data = {}
        datfile = dat_to_spmat(s['spike_path'],size=[256,256])
        # h5file = h5py.File(s['spike_path'])
        data['spike'] = np.array(datfile).astype(np.float32)
        data['flows'] = [read_gen(s['flow{:d}_path'.format(ii)]).astype(np.float32) for ii in range(1,3)]
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data


def get_test_datasets(cfg, valid=False, crop_len=500):
    test_datasets = {}
    for sd in TEST_SUBDATASET:
        # print(sd)
        cur_dataset = SpkLoader_test(cfg, sub_dataset=sd, valid=valid, crop_len=crop_len)
        test_datasets[sd] = cur_dataset
    return test_datasets
