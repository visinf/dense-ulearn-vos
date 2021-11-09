"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
import torch

from PIL import Image

import numpy as np
import torch.utils.data as data
import torchvision.transforms as tf

from .dataloader_base import DLBase

class DataSeg(DLBase):

    def __init__(self, cfg, split, ignore_labels=[], \
                 root=os.path.expanduser('./data'), renorm=False):

        super(DataSeg, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.ignore_labels = ignore_labels

        self._init_palette(cfg.DATASET.NUM_CLASSES)

        # train/val/test splits are pre-cut
        split_fn = os.path.join(self.root, "filelists", self.split + ".txt")
        assert os.path.isfile(split_fn)

        self.sequence_ids = []
        self.sequence_names = []
        def add_sequence(name):
            vlen = len(self.images)
            assert vlen >= cfg.DATASET.VIDEO_LEN, \
                "Detected video shorter [{}] than training length [{}]".format(vlen, \
                                                                cfg.DATASET.VIDEO_LEN)
            self.sequence_ids.append(vlen)
            self.sequence_names.append(name)
            return vlen

        self.images = []
        self.masks = []

        token = None
        with open(split_fn, "r") as lines:
            for line in lines:
                _image = line.strip("\n").split(' ')

                _mask = None
                if len(_image) == 2:
                    _image, _mask = _image
                else:
                    assert len(_image) == 1
                    _image = _image[0]

                _image = os.path.join(cfg.DATASET.ROOT, _image.lstrip('/'))
                assert os.path.isfile(_image), '%s not found' % _image
                self.images.append(_image)

                # each sequence may have a different length
                # do some book-keeping e.g. to ensure we have
                # sequences long enough for subsequent sampling
                _token = _image.split("/")[-2] # parent directory
                
                # sequence ID is in the filename
                #_token = os.path.basename(_image).split("_")[0] 
                if token != _token:
                    if not token is None:
                        add_sequence(token)
                    token = _token

                if _mask is None:
                    self.masks.append(None)
                else:
                    _mask = os.path.join(cfg.DATASET.ROOT, _mask.lstrip('/'))
                    assert os.path.isfile(_mask), '%s not found' % _mask
                    self.masks.append(_mask)

        # update the last sequence
        # returns the total amount of frames
        add_sequence(token)
        print("Loaded {} sequences".format(len(self.sequence_ids)))

        # definint data augmentation:
        print("Dataloader: {}".format(split), " #", len(self.images))
        print("\t {}: no augmentation".format(split))

        self.tf = tf.Compose([tf.ToTensor(), tf.Normalize(mean=self.MEAN, std=self.STD)])
        self._num_samples = len(self.images)

    def __len__(self):
        return len(self.sequence_ids)

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image
    
    def _mask2tensor(self, mask, num_classes=6):
        h,w = mask.shape
        ones = torch.ones(1,h,w)
        zeros = torch.zeros(num_classes,h,w)
        
        max_idx = mask.max()
        assert max_idx < num_classes, "{} >= {}".format(max_idx, num_classes)
        return zeros.scatter(0, mask[None, ...], ones)

    def __getitem__(self, index):

        seq_to = self.sequence_ids[index] - 1
        seq_from = 0 if index == 0 else self.sequence_ids[index-1] - 1

        images, masks = [], []
        n_obj = 0
        for _id_ in range(seq_from, seq_to):

            image = Image.open(self.images[_id_]).convert('RGB')

            if self.masks[_id_] is None:
                mask = Image.new('L', image.size)
            else:
                mask = Image.open(self.masks[_id_]) #.convert('L')

            image = self.tf(image)
            images.append(image)

            mask = torch.from_numpy(np.array(mask, np.long, copy=False))
            n_obj = max(n_obj, mask.max().item())
            masks.append(self._mask2tensor(mask))

        images = torch.stack(images, 0)
        masks = torch.stack(masks, 0)
        n_obj = torch.LongTensor([n_obj + 1]) # +1 background
        seq_name = self.sequence_names[index]

        return images, masks, n_obj, seq_name
