"""
Based on Jabri et al., (2020)
Credit: https://github.com/ajabri/videowalk.git
License: MIT
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models.resnet as torch_resnet
from torchvision.models.resnet import BasicBlock

class ResNet(torch_resnet.ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)

    def filter_layers(self, x):
        return [l for l in x if getattr(self, l) is not None]

    def remove_layers(self, remove_layers=[]):
        # Remove extraneous layers
        remove_layers += ['fc', 'avgpool']
        for layer in self.filter_layers(remove_layers):
            setattr(self, layer, None)

    def modify(self):

        # Set stride of layer3 and layer 4 to 1 (from 2)
        for layer in self.filter_layers(['layer3']):
            for m in getattr(self, layer).modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.stride = (1, 1)

        for layer in self.filter_layers(['layer4']):
            for m in getattr(self, layer).modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.stride = (1, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x if self.maxpool is None else self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x) 
        x4 = self.layer4(x3)

        return x4, x3

def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(pretrained='', remove_layers=[], train=True, **kwargs):

    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, **kwargs)
    model.modify()

    model.remove_layers(remove_layers)
    setattr(model, "fdim", 512)
    return model
