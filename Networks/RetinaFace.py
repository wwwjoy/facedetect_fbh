# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from BackBone import ResNet, resnet_spec
from FPN_SSH import FPN, SSH

class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        len = int(out.shape[1] * out.shape[2] * out.shape[3]) // 2
        out = out.reshape(int(out.shape[0]), len, 2)
        return out

class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        len = int(out.shape[1] * out.shape[2] * out.shape[3]) // 4
        out = out.reshape(int(out.shape[0]), len, 4)
        return out

class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        len = int(out.shape[1] * out.shape[2] * out.shape[3]) // 10
        out = out.reshape(int(out.shape[0]), len, 10)
        return out

class RetinaFace(nn.Module):
    def __init__(self, layers):
        """
        整体的检测网络
        :param layers: 18,50
        """
        super(RetinaFace, self).__init__()
        self.block, self.layer_list, self.in_channels_list = resnet_spec[layers]
        self.out_channels = 64
        self.layers = ResNet(self.block, self.layer_list)
        self.fpn = FPN(self.in_channels_list, self.out_channels)
        self.ssh1 = SSH(self.out_channels, self.out_channels)
        self.ssh2 = SSH(self.out_channels, self.out_channels)
        self.ssh3 = SSH(self.out_channels, self.out_channels)
        self.ssh4 = SSH(self.out_channels, self.out_channels)
        self.ClassHead = self._make_class_head(fpn_num=4, inchannels=self.out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=4, inchannels=self.out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=4, inchannels=self.out_channels)

    def _make_class_head(self, fpn_num=4, inchannels=64, anchor_num=3):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=4, inchannels=64, anchor_num=3):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=4, inchannels=64, anchor_num=3):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.layers(inputs)
        fpn = self.fpn(out)
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        feature4 = self.ssh4(fpn[3])
        features = [feature4, feature3, feature2, feature1]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        return bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions