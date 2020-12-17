# -*- coding: utf-8 -*-
"""
Function: Tools
Author: Wujia
Create Time: 2020/8/18 13:48
"""
import torch
import numpy as np
from math import sqrt as sqrt
import cv2

def no_deform_resize_pad(image, size):
    """
    不变形缩放图像，并返回缩放比例
    :param image:
    :param size:
    :return:
    """
    height, width = image.shape[0:2]
    long_side = max(width, height)
    resize_prop = size / float(long_side)
    inp_image = cv2.resize(image, None, None, fx=resize_prop, fy=resize_prop)
    h, w = inp_image.shape[0:2]
    image_t = cv2.copyMakeBorder(inp_image, 0, size - h, 0, size - w, cv2.BORDER_REPLICATE)
    return image_t, 1 / resize_prop

def rgb_mean_gpu(image, mean, device):
    """gpu计算图像rgb通道均一化"""
    image = torch.from_numpy(image.astype(np.float32))
    image = image.to(device)
    image = image - mean
    return image

class PriorBox(object):
    def __init__(self, box_specs_list, base_anchor_size):
        """
        生成anchors
        :param box_specs_list: anchor比例
        :param base_anchor_size: anchor尺寸
        """
        super(PriorBox, self).__init__()
        self._box_specs = box_specs_list
        self._base_anchor_size = base_anchor_size
        self._scales = []
        self._aspect_ratios = []
        for box_spec in self._box_specs:
            scales, aspect_ratios = zip(*box_spec)
            self._scales.append(scales)
            self._aspect_ratios.append(aspect_ratios)

    def num_anchors_per_location(self):
        return [len(box_specs) for box_specs in self._box_specs]

    def generate(self, feature_map_shape_list, im_height, im_width):
        im_height=float(im_height)
        im_width=float(im_width)
        anchor_strides = [(1.0 / pair[0], 1.0 / pair[1]) for pair in feature_map_shape_list]
        anchor_offsets = [(0.5 * stride[0],0.5 * stride[1]) for stride in anchor_strides]

        min_im_shape = min(im_height, im_width)
        scale_height = min_im_shape / im_height
        scale_width = min_im_shape / im_width
        base_anchor_size = [scale_height * self._base_anchor_size[0],
                            scale_width * self._base_anchor_size[1]]
        anchor_grid_list = []
        for grid_size, scales, aspect_ratios, stride, offset in zip(feature_map_shape_list,
                                                                    self._scales, self._aspect_ratios,
                                                                    anchor_strides, anchor_offsets):
            anchor_grid_list.append(self.tile_anchors(grid_height=grid_size[0],
                                                      grid_width=grid_size[1],
                                                      scales=scales,
                                                      aspect_ratios=aspect_ratios,
                                                      base_anchor_size=base_anchor_size,
                                                      anchor_stride=stride,
                                                      anchor_offset=offset))

        anchors= np.concatenate(np.array(anchor_grid_list),0)
        output = torch.Tensor(anchors)
        return output

    def tile_anchors(self,grid_height,
                          grid_width,
                          scales,
                          aspect_ratios,
                          base_anchor_size,
                          anchor_stride,
                          anchor_offset):
        ratio_sqrts = [sqrt(aspect_ratio) for aspect_ratio in aspect_ratios]
        heights = [scale / ratio_sqrt * base_anchor_size[0] for scale, ratio_sqrt in zip(scales, ratio_sqrts)]
        widths = [scale * ratio_sqrt * base_anchor_size[1] for scale, ratio_sqrt in zip(scales, ratio_sqrts)]
        # Get a grid of box centers
        y_centers = [float(y) * anchor_stride[0] + anchor_offset[0] for y in range(grid_height)]
        x_centers = [float(x) * anchor_stride[1] + anchor_offset[1] for x in range(grid_width)]
        x_centers, y_centers = self.meshgrid(x_centers, y_centers)
        widths_grid, x_centers_grid = self.meshgrid(widths, x_centers)
        heights_grid, y_centers_grid = self.meshgrid(heights, y_centers)
        bbox_centers = np.stack([x_centers_grid, y_centers_grid], axis=3)
        bbox_sizes = np.stack([widths_grid, heights_grid], axis=3)
        bbox_centers = np.reshape(bbox_centers, [-1, 2])
        bbox_sizes = np.reshape(bbox_sizes, [-1, 2])
        bbox_corners = np.concatenate([bbox_centers, bbox_sizes], 1)
        return bbox_corners  # [xc,yc,w,h]

    def meshgrid(self, x, y):
        x = np.array(x)
        y = np.array(y)
        x_exp = np.expand_dims(x, 0)
        y_exp = np.expand_dims(y, len(y.shape))
        x_exp_shape = x_exp.shape
        y_exp_shape = y_exp.shape
        xgrid = np.tile(x_exp, y_exp_shape)
        ygrid = np.tile(y_exp, x_exp_shape)
        return xgrid, ygrid

class Decode(object):
    def __init__(self, priors, variances):
        """
        v1解码
        :param priors: anchors
        :param variances: 缩放比例
        """
        super(Decode, self).__init__()
        self.priors = priors
        self.variances = variances

    def decode_bbox(self, pre_box):
        boxes = torch.cat((
            self.priors[:, :2] + pre_box[:, :2] * self.variances[0] * self.priors[:, 2:],
            self.priors[:, 2:] * torch.exp(pre_box[:, 2:] * self.variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landm(self, pre_landm):
        xcenter_a, ycenter_a, wa, ha = torch.unbind(self.priors.transpose(0, 1))
        landmarks = pre_landm.reshape(-1, 10).transpose(0, 1)
        tiled_anchor_center = torch.stack([xcenter_a, ycenter_a]).repeat(5, 1)
        tiled_anchor_size = torch.stack([wa, ha]).repeat(5, 1)
        tiled_variances = torch.stack([self.variances[0]]).repeat(10, 1)
        landms = (tiled_anchor_center + landmarks * tiled_variances * tiled_anchor_size).transpose(0, 1)
        return landms
