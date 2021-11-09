"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import matplotlib.cm as cm
import numpy as np
from PIL import ImagePalette

def colormap(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'uint8'
    cmap = []
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap.append((r, g, b))

    return cmap

def apply_cmap(masks_pred, cmap):
    canvas = np.zeros((masks_pred.shape[0], masks_pred.shape[1], 3))

    for label in np.unique(masks_pred):
        canvas[masks_pred == label, :] = cmap[label]

    return canvas #np.transpose(canvas, [2,0,1])


def create_palette(colormap, num):

    cmap = cm.get_cmap(colormap)
    palette = ImagePalette.ImagePalette()

    for n in range(num):
        val = n / num
        rgb = [int(255*x) for x in cmap(val)[:-1]]
        palette.getcolor(tuple(rgb))

    return palette

def custom_palette(nclasses, cname="rainbow"):
    cmap = cm.get_cmap(cname, nclasses)
    return cmap
