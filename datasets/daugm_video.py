"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import random
import numpy as np
import torch
from PIL import Image

import torchvision.transforms as tf
import torchvision.transforms.functional as F

class Compose:
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, args, *more_args):

        # allow for intermediate representations
        for t in self.segtransform:
            result = t(args, *more_args)
            args = result[0]
            more_args = result[1:]

        return result

class ToTensorMask:

    def __toByteTensor(self, pic):
        return torch.from_numpy(np.array(pic, np.int32, copy=False))

    def __call__(self, images, masks):

        new_masks = []
        for i, (image, mask) in enumerate(zip(images, masks)):
            images[i] = F.to_tensor(image)
            new_masks.append(self.__toByteTensor(mask))

        return images, new_masks

class CreateMask:
    """Create mask to hold invalid pixels
    (e.g. from rotations or downscaling)
    """

    def __call__(self, images):
        
        masks = []
        for i, image in enumerate(images):
            masks.append(Image.new("L", image.size))

        return images, masks

class Normalize:

    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):

        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)

        self.mean = mean
        self.std = std

    def __call__(self, images, masks):

        for i, (image, mask) in enumerate(zip(images, masks)):

            if self.std is None:
                for t, m in zip(image, self.mean):
                    t.sub_(m)
            else:
                for t, m, s in zip(image, self.mean, self.std):
                    t.sub_(m).div_(s)

        return images, masks

class ApplyMask:

    def __init__(self, ignore_label):
        self.ignore_label = ignore_label

    def __call__(self, images, masks):

        for i, (image, mask) in enumerate(zip(images, masks)):
            mask = mask > 0.
            images[i] *= (1. - mask.type_as(image))

        return images

class GuidedRandHFlip:

    def __call__(self, images, mask, affine=None):

        if affine is None:
            affine = [[0.,0.,0.,1.,1.] for _ in images]

        if random.random() > 0.5:
            for i, image in enumerate(images):
                affine[i][4] *= -1
                images[i] = F.hflip(image)

        return images, mask, affine

class AffineIdentity(object):

    def __call__(self, images, masks, affine=None):

        if affine is None:
            affine = [[0.,0.,0.,1.,1.] for _ in images]

        return images, masks, affine

class MaskRandScaleCrop(object):

    def __init__(self, scale_from, scale_to):
        self.scale_from = scale_from
        self.scale_to = scale_to
        #assert scale_from >= 1., "Zooming in is not supported yet"

    def get_scale(self):
        return random.uniform(self.scale_from, self.scale_to)

    def get_params(self, h, w, new_scale):
        # generating random crop
        # preserves aspect ratio
        new_h = int(new_scale * h)
        new_w = int(new_scale * w)

        # generating 
        if new_scale <= 1.:
            assert w >= new_w and h >= new_h, "{} vs. {} | {} / {}".format(w, new_w, h, new_h)
            i = random.randint(0, h - new_h)
            j = random.randint(0, w - new_w)
        else:
            assert w <= new_w and h <= new_h, "{} vs. {} | {} / {}".format(w, new_w, h, new_h)
            i = random.randint(h - new_h, 0)
            j = random.randint(w - new_w, 0)

        return i, j, new_h, new_w

    def __call__(self, images, masks, affine=None):

        if affine is None:
            affine = [[0.,0.,0.,1.,1.] for _ in images]

        W, H = images[0].size

        i2 = H / 2
        j2 = W / 2

        masks_new = []
        
        # one crop for all
        s = self.get_scale()

        ii, jj, h, w = self.get_params(H, W, s)

        # displacement of the centre
        dy = ii + h / 2 - i2
        dx = jj + w / 2 - j2

        for k, image in enumerate(images):

            affine[k][0] = dy
            affine[k][1] = dx
            affine[k][3] = 1 / s # scale

            if s <= 1.:
                assert ii >= 0 and jj >= 0
                # zooming in
                image_crop = F.crop(image, ii, jj, h, w)
                images[k] = image_crop.resize((W, H), Image.BILINEAR)

                mask_crop = F.crop(masks[k], ii, jj, h, w)
                masks_new.append(mask_crop.resize((W, H), Image.NEAREST))
            else:
                assert ii <= 0 and jj <= 0
                # zooming out
                pad_l = abs(jj)
                pad_r = w - W - pad_l
                pad_t = abs(ii)
                pad_b = h - H - pad_t

                image_pad = F.pad(image, (pad_l, pad_t, pad_r, pad_b))
                images[k] = image_pad.resize((W, H), Image.BILINEAR)

                mask_pad = F.pad(masks[k], (pad_l, pad_t, pad_r, pad_b), 1)
                masks_new.append(mask_pad.resize((W, H), Image.NEAREST))

        return images, masks, affine

class MaskScaleSmallest(object):

    def __init__(self, smallest_range):
        self.size = smallest_range

    def __call__(self, images, masks):
        assert len(images) > 0, "Non-empty array expected"

        new_size = self.size[0] + int((self.size[1] - self.size[0]) * random.random())

        w, h = images[0].size
        aspect = w / h

        if aspect > 1:
            new_h = new_size
            new_w = int(new_size * aspect)
        else:
            new_w = new_size
            new_h = int(new_size / aspect)

        new_size = (new_w, new_h)

        for i, (image, mask) in enumerate(zip(images, masks)):
            assert image.size == mask.size
            assert image.size == (w, h)

            images[i] = image.resize(new_size, Image.BILINEAR)
            masks[i] = mask.resize(new_size, Image.NEAREST)

        return images, masks

class MaskRandCrop:

    def __init__(self, size, pad_if_needed=False):
        self.size = size # (h, w)
        self.pad_if_needed = pad_if_needed

    def __pad(self, img, padding_mode='constant', fill=0):

        # pad the width if needed
        pad_width = self.size[1] - img.size[0]
        pad_height = self.size[0] - img.size[1]
        if self.pad_if_needed and (pad_width > 0 or pad_height > 0):
            pad_l = max(0, pad_width // 2)
            pad_r = max(0, pad_width - pad_l)
            pad_t = max(0, pad_height // 2)
            pad_b = max(0, pad_height - pad_t)
            img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), fill, padding_mode)

        return img

    def __call__(self, images, masks):

        for i, (image, mask) in enumerate(zip(images, masks)):
            images[i] = self.__pad(image)
            masks[i] = self.__pad(mask, fill=1)

        i, j, h, w = tf.RandomCrop.get_params(images[0], self.size)

        for k, (image, mask) in enumerate(zip(images, masks)):
            images[k] = F.crop(image, i, j, h, w)
            masks[k] = F.crop(mask, i, j, h, w)

        return images, masks

class MaskCenterCrop:

    def __init__(self, size):
        self.size = size # (h, w)

    def __call__(self, images, masks):

        for i, (image, mask) in enumerate(zip(images, masks)):
            images[i] = F.center_crop(image, self.size)
            masks[i] = F.center_crop(mask, self.size)

        return images, masks

class MaskRandHFlip:

    def __call__(self, images, masks):

        if random.random() > 0.5:

            for i, (image, mask) in enumerate(zip(images, masks)):
                images[i] = F.hflip(image)
                masks[i] = F.hflip(mask)

        return images, masks
