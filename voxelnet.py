# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import Tuple

# from torch import Tensor

# from mmdet3d.registry import MODELS
# from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
# from .single_stage import SingleStage3DDetector

# @MODELS.register_module()
# class VoxelNet(SingleStage3DDetector):
#     r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

#     def __init__(self,
#                  voxel_encoder: ConfigType,
#                  middle_encoder: ConfigType,
#                  backbone: ConfigType,
#                  neck: OptConfigType = None,
#                  bbox_head: OptConfigType = None,
#                  train_cfg: OptConfigType = None,
#                  test_cfg: OptConfigType = None,
#                  data_preprocessor: OptConfigType = None,
#                  init_cfg: OptMultiConfig = None) -> None:
#         super().__init__(
#             backbone=backbone,
#             neck=neck,
#             bbox_head=bbox_head,
#             train_cfg=train_cfg,
#             test_cfg=test_cfg,
#             data_preprocessor=data_preprocessor,
#             init_cfg=init_cfg)
#         self.voxel_encoder = MODELS.build(voxel_encoder)
#         self.middle_encoder = MODELS.build(middle_encoder)

#     def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
#         """Extract features from points."""
#         voxel_dict = batch_inputs_dict['voxels']
#         voxel_features = self.voxel_encoder(voxel_dict['voxels'],
#                                             voxel_dict['num_points'],
#                                             voxel_dict['coors'])
#         batch_size = voxel_dict['coors'][-1, 0].item() + 1
#         x = self.middle_encoder(voxel_features, voxel_dict['coors'],
#                                 batch_size)
#         x = self.backbone(x)
        
#         if self.with_neck:
#             x = self.neck(x)
#         return x







# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStage3DDetector

import torch
import torch.nn as nn
from torch.nn import Sequential as Sequential
from mmcv.ops import SparseConvTensor

class CA_Attention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, ratio=4):
        super(CA_Attention, self).__init__()
        #平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        #MLP  除以16是降维系数
        self.fc1   = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False) #kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        # self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        out = avg_out + max_out
        weight_pillars = self.sigmoid(out)
        out = weight_pillars*x

        # out = self.conv2d(out)


        # out = self.Conv(out)
        return out

class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
        
    def forward(self, x):
        return x * self.scale

class AttentionConv2D(nn.Module):
    def __init__(self,
              in_channels=64,
              out_channels=64,
              ):
        super(AttentionConv2D, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels


        # conv a and b
        self._conv_a = Sequential(
            nn.Conv2d(
            self._in_channels, int(self._in_channels / 2), kernel_size=3,
            stride=1, padding=1)
        )

        self._conv_b = Sequential(
            nn.Conv2d(
            self._in_channels, int(self._in_channels / 2), kernel_size=3,
            stride=1, padding=1)
        )

        # attention a and b
        self.conv_1_1 = Sequential(
            nn.Conv2d(in_channels=self._in_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
            )

        self._conv_attention_a = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

        # self._conv_attention_b = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.init_attention_weight()
        self.scale_H = Scale(scale=1.0)
        self.scale_P = Scale(scale=1.0)
        self._sigmoid = nn.Sigmoid()

    
    def init_attention_weight(self):
    # [c_out, c_in, kernel_size[0],kernel_size[1]]
    # TODO(Xuetao): hard code 3 for conv 1*1 + mean + max of input.
        with torch.no_grad():
            self._conv_attention_a.weight.data[:, 3:, :, :] = torch.abs(self._conv_attention_a.weight.data[:, 3:, :, :])
            # self._conv_attention_b.weight.data[:, 3:, :, :] = torch.abs(self._conv_attention_b.weight.data[:, 3:, :, :])
        
    def forward(self, Height, Pillars):

        Height_x = self._conv_a(Height)
        Pillars_x = self._conv_b(Pillars)

        Cat_HP = torch.cat([Height_x, Pillars_x], dim=1)
        Cat_attention = self.conv_1_1(Cat_HP)

        Fusion_cat = torch.cat(
            [Cat_attention, 
            torch.max(Cat_HP, 1)[0].unsqueeze(1), 
            torch.mean(Cat_HP, 1).unsqueeze(1)], dim=1)

        attention_maps_H = self._conv_attention_a(Fusion_cat)
        attention_maps_P = self._conv_attention_a(Fusion_cat)

        Height_x = (1 + self.scale_H(self._sigmoid(attention_maps_H))) * Height_x
        Pillars_x = (1 + self.scale_P(self._sigmoid(attention_maps_P))) * Pillars_x
        out = torch.cat((Height_x, Pillars_x), dim=1)
        return out


class backbone_H(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(backbone_H, self).__init__()
        #平均池化
        self.downsample_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.downsample_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.downsample_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
    def forward(self, x):
        out_1 = self.downsample_1(x)
        out_2 = self.downsample_2(out_1)
        out_3 = self.downsample_3(out_2)
        out = tuple([out_1, out_2, out_3])
        return out


@MODELS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: ConfigType,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

        self.AttentionConv2D_64 = AttentionConv2D(in_channels=64, out_channels=64).cuda()
        self.AttentionConv2D_128 = AttentionConv2D(in_channels=128, out_channels=128).cuda()
        self.AttentionConv2D_256 = AttentionConv2D(in_channels=256, out_channels=256).cuda()
        self.backbone_H = backbone_H().cuda()
        self.sparse_shape = [64, 496, 432]
        self.num_features = 4
    
    def extract_Hight_feature(self, voxel_dict, batch_size):
        features = voxel_dict['voxels_h']
        num_points = voxel_dict['num_points_h']
        # points_mean = features[:, :, :self.num_features].sum(dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        
        masked_tensor = features[:, :, 2].clone()  # 防止修改原始张量
        masked_tensor[masked_tensor == 0] = float('-inf')
        points_z_max = masked_tensor.max(dim=1, keepdim=False)[0]
        points_z_max[points_z_max == float('-inf')] = 0
        
        points_z_max = points_z_max.reshape(-1, 1)
        voxel_features = points_z_max.contiguous()
        coors = voxel_dict['coors_h']
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        features = input_sp_tensor.dense()
        return features.reshape(batch_size, 64, 496, 432)

    def Share_fusion_wo_CA(self, Height_feature, x):
        out_0 = self.AttentionConv2D_64(Height_feature[0], x[0])
        # out_0 = self.CA_Attention_64(out_0)
        out_1 = self.AttentionConv2D_128(Height_feature[1], x[1])
        # out_1 = self.CA_Attention_128(out_1)
        out_2 = self.AttentionConv2D_256(Height_feature[2], x[2])
        # out_2 = self.CA_Attention_256(out_2)
        x = tuple([out_0, out_1, out_2])
        return x
    

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)
        x = self.backbone(x)

        Height_features = self.extract_Hight_feature(voxel_dict, batch_size)
        Height_features = self.backbone_H(Height_features)
        x = self.Share_fusion_wo_CA(Height_features, x)


        if self.with_neck:
            x = self.neck(x)
        return x