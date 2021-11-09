"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

from .resnet18 import resnet18
from .net import Net
from .framework import Framework

def get_model(cfg, *args, **kwargs):

    backbones = {
        'resnet18': resnet18
    }

    def create_net():
        backbone = backbones[cfg.MODEL.ARCH.lower()](*args, **kwargs)
        return Net(cfg, backbone)

    net = create_net()
    return Framework(cfg, net)
