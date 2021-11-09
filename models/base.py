"""
Base class for network models
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):

    _trainable = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.SyncBatchNorm)
    _batchnorm = (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)

    def __init__(self):
        super().__init__()
        # we may want a different learning rate
        # for new layers
        self.from_scratch_layers = []

        # we may want to freeze some layers
        self.not_training = []

        # we may want to freeze BN (means/stds only)
        self.bn_freeze = []

    def lr_mult(self):
        """Learning rate multiplier for weights.
        Returns: [old, new]"""
        return 1., 1.

    def lr_mult_bias(self):
        """Learning rate multiplier for bias.
        Returns: [old, new]"""
        return 2., 2.

    def _is_learnable(self, layer):
         return isinstance(layer, BaseNet._trainable)

    def _from_scratch(self, net, ignore=[]):

        for layer in net.modules():
            if self._is_learnable(layer):
                self.from_scratch_layers.append(layer)

    def _freeze_bn(self, net, ignore=[]):
        """Add layers to use in .eval() mode only"""

        for layer in net.modules():
            if isinstance(layer, BaseNet._batchnorm) and \
                    not layer in ignore:

                assert hasattr(layer, "eval") and callable(layer.eval)
                print("Freezing ", layer)
                self.bn_freeze.append(layer)

        print("Frozen BN: ", len(self.bn_freeze))

    def _fix_bn(self, layer):
        if isinstance(layer, nn.BatchNorm2d):
            self.not_training.append(layer)

        elif isinstance(layer, nn.Module):
            for c in layer.children():
                self._fix_bn(c)

    def __set_grad_mode(self, layer, mode, only_type=None):

        if hasattr(layer, "weight"):
            if only_type is None or isinstance(layer, only_type):
                layer.weight.requires_grad = mode

        if hasattr(layer, "bias") and not layer.bias is None:
            if only_type is None or isinstance(layer, only_type):
                layer.bias.requires_grad = mode

        if isinstance(layer, nn.Module):
            for c in layer.children():
                self.__set_grad_mode(c, mode)

    def train(self, mode=True):
        super().train(mode)

        # some layers have to be frozen
        for layer in self.not_training:
            self.__set_grad_mode(layer, False)

        for layer in self.bn_freeze:
            assert hasattr(layer, "eval") and callable(layer.eval)
            layer.eval()

    def parameter_groups(self, base_lr, wd):

        w_old, w_new = self.lr_mult()
        b_old, b_new = self.lr_mult_bias()

        groups = ({"params": [], "weight_decay":  wd, "lr": w_old*base_lr}, # weight learning
                  {"params": [], "weight_decay": 0.0, "lr": b_old*base_lr}, # bias learning
                  {"params": [], "weight_decay":  wd, "lr": w_new*base_lr}, # weight finetuning
                  {"params": [], "weight_decay": 0.0, "lr": b_new*base_lr}) # bias finetuning

        for m in self.modules():

            if not self._is_learnable(m):
                if hasattr(m, "weight") or hasattr(m, "bias"):
                    print("Skipping layer with parameters: ", m)
                continue

            if not m.weight is None and m.weight.requires_grad:
                if m in self.from_scratch_layers:
                    groups[2]["params"].append(m.weight)
                else:
                    groups[0]["params"].append(m.weight)
            elif not m.weight is None:
                print("Skipping W: ", m, m.weight.size())

            if m.bias is not None and m.bias.requires_grad:
                if m in self.from_scratch_layers:
                    groups[3]["params"].append(m.bias)
                else:
                    groups[1]["params"].append(m.bias)
            elif m.bias is not None:
                print("Skipping b: ", m, m.bias.size())

        return groups
    
    @staticmethod
    def _resize_as(x, y):
        return F.interpolate(x, y.size()[-2:], mode="bilinear", align_corners=True)
