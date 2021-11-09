"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
import torch

from PIL import Image

import numpy as np
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
        self._init_palette(self.cfg.DATASET.NUM_CLASSES)

        # train/val/test splits are pre-cut
        split_fn = os.path.join(self.root, self.split + ".txt")
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
        self.flags = []

        token = None
        with open(split_fn, "r") as lines:
            for line in lines:
                _flag, _image, _mask = line.strip("\n").split(' ')

                # save every frame
                #_flag = 1
                self.flags.append(int(_flag))

                _image = os.path.join(cfg.DATASET.ROOT, _image.lstrip('/'))
                assert os.path.isfile(_image), '%s not found' % _image

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

                self.images.append(_image)

                if _mask is None:
                    self.masks.append(None)
                else:
                    _mask = os.path.join(cfg.DATASET.ROOT, _mask.lstrip('/'))
                    #assert os.path.isfile(_mask), '%s not found' % _mask
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

    
    def _mask2tensor(self, mask, num_classes=6):
        h,w = mask.shape
        ones = torch.ones(1,h,w)
        zeros = torch.zeros(num_classes,h,w)
        
        max_idx = mask.max()
        assert max_idx < num_classes, "{} >= {}".format(max_idx, num_classes)
        return zeros.scatter(0, mask[None, ...], ones)
    
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


    def __getitem__(self, index):
        
        seq_to = self.sequence_ids[index]
        seq_from = 0 if index == 0 else self.sequence_ids[index - 1]

        image0 = Image.open(self.images[seq_from])
        w,h = image0.size

        images, masks, fns, flags = [], [], [], []
        tracks = torch.LongTensor(self.cfg.DATASET.NUM_CLASSES).fill_(-1)
        masks = torch.LongTensor(self.cfg.DATASET.NUM_CLASSES, h, w).zero_()
        known_ids = set()

        for t in range(seq_from, seq_to):

            t0 = t - seq_from
            image = Image.open(self.images[t]).convert('RGB')

            fns.append(os.path.basename(self.images[t].replace(".jpg", "")))
            flags.append(self.flags[t])

            if os.path.isfile(self.masks[t]):
                mask = Image.open(self.masks[t])
                mask = torch.from_numpy(np.array(mask, np.long, copy=False))

                unique_ids = np.unique(mask)
                for oid in unique_ids:
                    if not oid in known_ids:
                        tracks[oid] = t0
                        known_ids.add(oid)
                        masks[oid] = (mask == oid).long()
            else:
                mask = Image.new('L', image.size)

            image = self.tf(image)
            images.append(image)

        images = torch.stack(images, 0)
        seq_name = self.sequence_names[index]
        flags = torch.LongTensor(flags)

        return images, images, masks, tracks, len(known_ids), fns, flags, seq_name
