"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import torch.utils.data as data
from utils.palette import custom_palette

class DLBase(data.Dataset):

    def __init__(self, *args, **kwargs):
        super(DLBase, self).__init__(*args, **kwargs)

        # RGB
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        self._init_means()

    def _init_means(self):
        self.MEAN255 = [255.*x for x in self.MEAN]
        self.STD255 = [255.*x for x in self.STD]

    def _init_palette(self, num_classes):
        self.palette = custom_palette(num_classes)

    def get_palette(self):
        return self.palette

    def remove_labels(self, mask):
        # Remove labels not in training
        for ignore_label in self.ignore_labels:
            mask[mask == ignore_label] = 255

        return mask.long()

