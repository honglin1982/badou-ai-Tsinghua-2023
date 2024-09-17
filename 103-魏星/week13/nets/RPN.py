# _*_ coding : utf-8 _*_
# @Time : 2023/8/4 17:37
# @Author : weixing
# @FileName : RPN
# @Project : cv

import torch
from torch import nn
from torch.nn import functional as F
from util.anchorsbox import generate_anchor_base, _enumerate_shifted_anchor
from util.bboxutil import loc2bbox
from config.Config import Config
from torchvision.ops import nms
from util.bboxutil import normal_init
import numpy as np

config = Config()

#
'''
RPN分为两路：rpn_cls和rpn_bbox


'''


class ProposalCreator():
    def __init__(self,
                 mode,
                 nms_thresh=0.7,
                 n_train_pre_nms=3000,
                 n_train_post_nms=300,
                 n_test_pre_nms=3000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.mode = mode
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        # 将RPN网络预测结果转化成建议框
        roi = loc2bbox(anchor, loc)

        # 利用slice进行分割，防止建议框超出图像边缘
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[1])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[0])

        # 宽高的最小值不可以小于16
        min_size = self.min_size * scale
        # 计算高宽
        ws = roi[:, 2] - roi[:, 0]
        hs = roi[:, 3] - roi[:, 1]
        # 防止建议框过小
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]
        # 取出成绩最好的一些建议框
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        roi = nms(roi, self.nms_thresh)
        roi = torch.Tensor(roi)
        roi = roi[:n_post_nms]
        return roi


'''
RPN主要分为两路：rpn_cls和rpn_bbox

'''


class RegionProposalNetwork(nn.Module):
    def __init__(
            self, in_channels=512, mid_channels=512, ratios=config.ratios,
            anchor_scales=config.anchor_scales, feat_stride=16,
            mode="training",
    ):
        super(RegionProposalNetwork, self).__init__()
        # 生成base_anchor
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)  # (9,4)
        # 步长，压缩的倍数
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(mode)
        # 每一个网格上默认先验框的数量
        n_anchor = self.anchor_base.shape[0]

        # 先进行一个3x3的卷积  # input [N,M/16,N/16,in_channels] out[N,M/16,N/16,mid_channels]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # 分类 预测先验框内部是否包含物体  rpn_cls  # input [N,M/16,N/16,mid_channels] out[N,M/16,N/16,18]
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # 回归 预测对先验框进行调整  rpn_bbox  # input [N,M/16,N/16,mid_channels] out[N,M/16,N/16,36]
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, height, width = x.shape
        # 对共享特征层进行一个3x3的卷积
        after_conv = F.relu(self.conv1(x))
        # 回归预测
        rpn_locs = self.loc(after_conv)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # 分类预测
        rpn_scores = self.score(after_conv)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        # 进行softmax
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        # 生成先验框
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, height, width)
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor
