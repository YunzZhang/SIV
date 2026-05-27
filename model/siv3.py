import torch
import torch.nn as nn
import torch.nn.functional as F
from .sr_refine import MultiResPIVRefiner
from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoderQuarter, GraphEncoder
from .utils import bilinear_sampler, coords_grid, upflow4
from .path_match import PathMatch
from.rep import DPHT
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class SIV(nn.Module):
    def __init__(self, args):
        super(SIV, self).__init__()
        self.args = args

        if 'max_offset' not in self.args:
            self.max_offset = 192
        if 'mixed_precision' not in self.args:
            self.mixed_precision = True
        if 'test_mode' not in self.args:
            self.test_mode = False
        self.max_offset = 192
        self.mixed_precision = True
        self.test_mode = False
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.dropout = 0
        self.rep = DPHT()

        # feature network, and update block
        # self.fnet = SparseEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
        self.fnet = BasicEncoderQuarter(output_dim=256, norm_fn='instance', dropout=self.dropout)
        self.cnet = GraphEncoder(args, 128)
        self.update_block_s = SmallUpdateBlock(hidden_dim=hdim)
        self.update_block = BasicUpdateBlock(hidden_dim=hdim)
        self.refiner = MultiResPIVRefiner()
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    ####################################################################################
    ## Tools functions for neural networks
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def num_parameters(self):
        return sum([p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])
    
    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def random_init_flow(self, fmap, max_offset, test_mode = False):
        N, C, H, W = fmap.shape
        if test_mode:
            init_seed = 20
            torch.manual_seed(init_seed)
            torch.cuda.manual_seed(init_seed)
        flow = (torch.rand(N, 2, H, W) - 0.5) * 2
        flow = flow.to(fmap.device) * max_offset


        return flow

    def upsample_flow(self, flow, mask, rate=4):
        """ Upsample flow field [H/rate, W/rate, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate * H, rate * W)

    def build_pyramid(self, fmap0, fmap1, cnet, max_layers = 5,  min_width = 40):
        py_fmap0 = []
        py_fmap1 = []        

        py_cnet = []
        py_fmap0.append(fmap0)
        py_fmap1.append(fmap1)

        py_cnet.append(cnet)

        curr_fmap0 = fmap0
        curr_fmap1 = fmap1

        curr_cnet = cnet
        for i in range(max_layers - 1):
            if (curr_fmap0.shape[2] < min_width) and (curr_fmap0.shape[3] < min_width):
                break
            down_scale = 2**(i + 1)
            curr_fmap0 = F.avg_pool2d(curr_fmap0, 2, stride=2)
            curr_fmap1 = F.avg_pool2d(curr_fmap1, 2, stride=2)

            curr_cnet = F.avg_pool2d(curr_cnet, 2, stride=2)
            py_fmap0.append(curr_fmap0)
            py_fmap1.append(curr_fmap1)

            py_cnet.append(curr_cnet)
        
        return py_fmap0, py_fmap1, py_cnet

    def upflow(self, flow, targetMap, mode='bilinear'):
        """ Upsample flow """
        new_size = (targetMap.shape[2], targetMap.shape[3])
        factor = 1.0 * targetMap.shape[2] / flow.shape[2]
        return  factor * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


 
    def forward(self, spks, iters = 12, init_flow = None, dt=21):
        """ Estimate optical flow between pair of frames """

        # if self.test_mode and init_flow is not None:
        #     flow_up = self.inference(spks, iters = iters, init_flow = init_flow)
        #     return flow_up
        if dt==21:
            seq1 = spks[:, 0:21].contiguous()
            seq2 = spks[:, 21:42].contiguous()
        elif dt==11:
            seq1 = spks[:, 5:26].contiguous()
            seq2 = spks[:, 16:37].contiguous()        
        else:
            raise ValueError("dt should be 21 or 11")    

        rep_list1 = self.rep(seq1)
        rep_list2 = self.rep(seq2)


        spk1 = rep_list1[0]
        spk2 = rep_list2[0]

 
        hdim = self.hidden_dim
        cdim = self.context_dim
        # run the feature network
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([spk1, spk2])     

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()


        # run the context network
        with autocast(enabled=self.mixed_precision):

            cnet = self.cnet(spk1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

            # 1/4 -> 1/16
            # feature
            s_fmap1 = F.avg_pool2d(fmap1, 4, stride=4)
            s_fmap2 = F.avg_pool2d(fmap2, 4, stride=4)

            # context(left)
            s_net = F.avg_pool2d(net, 4, stride=4)
            s_inp = F.avg_pool2d(inp, 4, stride=4)

        # 1/16
        s_patch_fn = PathMatch(s_fmap1, s_fmap2)


        # init flow
        s_flow = None
        s_flow = self.random_init_flow(s_fmap1, max_offset=self.max_offset // 16, test_mode = self.test_mode)
       
        # small initial: 1/16
        flow = None
        flow_1_16_list = []
        flow_1_4_list = []
        flow_predictions = []
        for itr in range(iters):
            # --------------- update1 ---------------
            s_flow = s_flow.detach()

            out_corrs= s_patch_fn(s_flow, is_search=False)

            # GRU Update
            with autocast(enabled=self.mixed_precision):
                # 4D net map to 2D dense vector
                s_net, up_mask, delta_flow = self.update_block_s(s_net, s_inp, out_corrs, s_flow)

            s_flow = s_flow + delta_flow


            # flow_up = []
            flow = self.upsample_flow(s_flow, up_mask, rate=4)

            flow_up = upflow4(flow)

            flow_predictions.append(flow_up)

            # --------------- update2 ---------------
            s_flow = s_flow.detach()

            out_corrs = s_patch_fn(s_flow, is_search=True)

            # GRU Update
            with autocast(enabled=self.mixed_precision):
                # 4D net map to 2D dense vector
                s_net, up_mask, delta_flow = self.update_block(s_net, s_inp, out_corrs, s_flow)

            s_flow = s_flow + delta_flow

            flow_1_16_list.append(s_flow)

            flow = self.upsample_flow(s_flow, up_mask, rate=4)

            flow_up = upflow4(flow)

            flow_predictions.append(flow_up)

        patch_fn = PathMatch(fmap1, fmap2)

        # large refine: 1/4
        for itr in range(iters):
            # --------------- update1 ---------------
            flow = flow.detach()

            out_corrs = patch_fn(flow, is_search=False)

            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block_s(net, inp, out_corrs, flow)

            flow = flow + delta_flow

            flow_up = self.upsample_flow(flow, up_mask, rate=4)

            flow_predictions.append(flow_up)

            # --------------- update2 ---------------
            flow = flow.detach()

            out_corrs = patch_fn(flow, is_search=True)


            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow

            flow_1_4_list.append(flow)

            flow_up = self.upsample_flow(flow, up_mask, rate=4)\
            
            # flow_refined = self.refiner(flow_up)
            
            flow_predictions.append(flow_up)

        # if self.test_mode:
        #     return flow_up
        multiscale_flow = [flow_1_16_list[-1], flow_1_4_list[-1], flow_predictions[-1]]

        flow_refined = self.refiner(multiscale_flow)

        flow_predictions.append(flow_refined)
        
        return flow_predictions
