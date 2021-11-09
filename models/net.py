"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseNet

class MLP(nn.Sequential):

    def __init__(self, n_in, n_out):
        super().__init__()

        self.add_module("conv1", nn.Conv2d(n_in, n_in, 1, 1))
        self.add_module("bn1", nn.BatchNorm2d(n_in))
        self.add_module("relu", nn.ReLU(True))
        self.add_module("conv2", nn.Conv2d(n_in, n_out, 1, 1))

class Net(BaseNet):

    def __init__(self, cfg, backbone):
        super(Net, self).__init__()

        self.cfg = cfg
        self.backbone = backbone
        self.emb_q = MLP(backbone.fdim, cfg.MODEL.FEATURE_DIM)

    def lr_mult(self):
        """Learning rate multiplier for weights.
        Returns: [old, new]"""
        return 1., 1.

    def lr_mult_bias(self):
        """Learning rate multiplier for bias.
        Returns: [old, new]"""
        return 2., 2.

    def forward(self, frames, norm=True):
        """Forward pass to extract projection and task features"""

        # extracting the time dimension
        res4, res3 = self.backbone(frames)

        # B,K,H,W
        query = self.emb_q(res4)

        if norm:
            query = F.normalize(query, p=2, dim=1)
            res3 = F.normalize(res3, p=2, dim=1)
            res4 = F.normalize(res4, p=2, dim=1)

        return query, res3, res4
