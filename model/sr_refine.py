import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossConv(nn.Module):
    """Lightweight module for coarse-scale guiding fine-scale"""
    def __init__(self, ch):
        super().__init__()
        self.modulator = nn.Sequential(
            nn.Conv2d(ch, ch//4, 1),  # Channel compression
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//4, ch, 3, padding=1),  # Spatial modulation
            nn.Sigmoid()
        )
        self.blender = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 2, 1),  # Expand channels
            nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch, 1)  # Compress back to original channels
        )

    def forward(self, feat_fine, feat_coarse_up):
        mod = self.modulator(feat_coarse_up)
        guided = feat_fine * mod
        fused = torch.cat([guided, feat_fine], dim=1)
        return self.blender(fused)

class MultiResPIVRefiner(nn.Module):
    def __init__(self, flow_channels=2, base_channels=32, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.base_channels = base_channels

        # Multi-scale feature extractors
        self.scale_processors = nn.ModuleList()
        for i in range(num_scales):
            kernel_size = 3 + 2 * i
            self.scale_processors.append(nn.Sequential(
                nn.Conv2d(flow_channels, base_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, 5, padding=2),
                nn.ReLU(inplace=True),
            ))

        # Cross-scale fusion modules (coarse -> fine)
        self.cross_convs = nn.ModuleList([
            CrossConv(base_channels) for _ in range(num_scales - 1)
        ])

        # Fuse all scale features
        self.final_fusion = nn.Sequential(
            nn.Conv2d(base_channels * num_scales, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # PIV quality assessment & output residual
        self.quality_assessment = nn.Sequential(
            nn.Conv2d(base_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.output_conv = nn.Conv2d(base_channels, flow_channels, 3, padding=1)

    def forward(self, multi_scale_flows):
        scale_feats = []
        # 1. Multi-scale feature extraction
        for i, (flow, processor) in enumerate(zip(multi_scale_flows, self.scale_processors)):
            scale_feats.append(processor(flow))

        # 2. Coarse-scale features guide fine-scale features (CrossConv)
        for i in range(self.num_scales - 1):
            feat_coarse = scale_feats[i]
            feat_fine = scale_feats[i + 1]

            feat_coarse_up = F.interpolate(
                feat_coarse, size=feat_fine.shape[2:], 
                mode='bilinear', align_corners=False
            )

            # Cross-scale fusion updates fine-scale features
            scale_feats[i + 1] = self.cross_convs[i](feat_fine, feat_coarse_up)

        # 3. Upsample all scale features to highest resolution
        target_size = multi_scale_flows[-1].shape[2:]
        full_scale_feats = []
        for feat in scale_feats:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            full_scale_feats.append(feat)

        # 4. Fuse features
        fused_feat = self.final_fusion(torch.cat(full_scale_feats, dim=1))

        # 5. Output quality map and residual flow
        quality_map = self.quality_assessment(fused_feat)
        residual = self.output_conv(fused_feat)

        # 6. Residual weighted refinement
        refined_flow = multi_scale_flows[-1] + quality_map * residual

        return refined_flow