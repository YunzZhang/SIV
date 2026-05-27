import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import *

class ResidualFeatureDownsamling(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualFeatureDownsamling, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=1,stride=1)

        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm3 = nn.Sequential()

    def forward(self, x):
        out1 = self.conv1(x)

        out2 = self.pool(x)
        out2 = self.conv2(out2)

        out = out1 + out2

        out = self.norm3(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1, RFD = False):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            # if not stride == 1:
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            # if not stride == 1:
            self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            # if not stride == 1:
            self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            # if not stride == 1:
            self.norm3 = nn.Sequential()

        # if RFD:
        #     self.downsample = ResidualFeatureDownsamling(in_planes, planes, norm_fn, stride=stride)
        
        # else:
        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(128, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class BasicEncoderQuarter(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoderQuarter, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(128, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=1)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
       
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
            batch = x.shape[0] // batch_dim
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim]*batch, dim=0)

        return x


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, alpha=0.2):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, out_dim * 2))  # concat approach
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, A=None):
        """
        x: [B, N, C] node features
        A: [B, N, N] original adjacency matrix (optional, can be omitted if using fully connected attention)
        """
        B, N, _ = x.size()
        h = self.W(x)  # [B, N, out_dim]
        
        # Construct (i,j) feature pairs: [B, N, N, 2*out_dim]
        h_i = h.unsqueeze(2).expand(B, N, N, -1)  # each node i
        h_j = h.unsqueeze(1).expand(B, N, N, -1)  # each neighbor j
        a_input = torch.cat([h_i, h_j], dim=-1)   # [B, N, N, 2*out_dim]

        # Calculate attention logits
        e = self.leakyrelu((a_input * self.attn).sum(dim=-1))  # [B, N, N]

        if A is not None:
            e = e.masked_fill(A == 0, float('-inf'))  # only keep original graph connections

        alpha = F.softmax(e, dim=-1)  # normalize attention
        alpha = self.dropout(alpha)

        h_prime = torch.bmm(alpha, h)  # [B, N, out_dim]
        return F.relu(h_prime)  # activation

class GATEncoder(nn.Module):
    def __init__(self, dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GATLayer(dim, dim, dropout=dropout))

    def forward(self, x, A=None):
        for gat in self.layers:
            x_res = x
            x = gat(x, A)
            x = x + x_res
        return x

class GraphEncoder(nn.Module):
    def __init__(self, args, chnn, k=128):
        super().__init__()
        self.C_cpr = nn.Sequential(
            nn.Conv2d(chnn, chnn, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chnn, k, 1))
        self.C_cad = nn.Conv1d(chnn, chnn, kernel_size=1)
        self.gcn = GATEncoder(chnn, num_layers=2)
        self.C_mpr = nn.Conv2d(chnn, k, kernel_size=1)
        self.C_ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(chnn, chnn//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chnn//4, chnn, 1))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.zero = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.resEncoder = BasicEncoderQuarter(output_dim=256, norm_fn='instance', dropout=args.dropout)
    def __func_cpr(self, x, C_cpr):
        b, c, h, w = x.shape
        mp = C_cpr(x).view(b, -1, h*w)
        vs = torch.einsum('b c n , b k n -> b c k', x.view(b, c, -1), mp)
        vs = F.normalize(vs, p=2, dim=1) 
        return vs, mp

    def __func_cg(self, vc, C_cad, gcn):
        vca = F.normalize(C_cad(vc), p=2, dim=1)
        A = torch.einsum('b c k , b c l -> b k l', vca, vca)
        vc_T = vc.permute(0, 2, 1).contiguous()
        vco = gcn(vc_T, A)
        return vco

    def forward(self, x):
        feat_ctx = x
        b, c, h, w = feat_ctx.shape

        vc, mp = self.__func_cpr(feat_ctx, self.C_cpr)
        vco = self.__func_cg(vc, self.C_cad, self.gcn)
        feat_ctxa = torch.bmm(vco, mp)
        feat_ctxa = feat_ctxa.view(b, -1, h, w)
        out_gcn = x + F.relu(feat_ctxa * self.alpha)
        out_cnn = self.resEncoder(out_gcn)
        return out_cnn