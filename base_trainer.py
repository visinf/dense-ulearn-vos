"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
import torch
import math

import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils

from torch.utils.tensorboard import SummaryWriter
from torch.optim.optimizer import Optimizer

from utils.checkpoints import Checkpoint
from utils.palette_davis import palette as palette_davis

from PIL import Image

from matplotlib import cm

class BaseTrainer(object):

    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.start_epoch = 0
        self.best_score = -1e16
        self.checkpoint = Checkpoint(args.snapshot_dir, max_n = 3)

        logdir = os.path.join(args.logdir, 'train')
        self.writer = SummaryWriter(logdir)

    def checkpoint_best(self, score, epoch, temp):

        if score > self.best_score:
            print(">>> Saving checkpoint with score {:3.2e}, epoch {}".format(score, epoch))
            self.best_score = score
            self.checkpoint.checkpoint(score, epoch, temp)

            return True

        return False

    @staticmethod
    def get_optim(params, cfg):

        if not hasattr(torch.optim, cfg.OPT):
            print("Optimiser {} not supported".format(cfg.OPT))
            raise NotImplementedError

        optim = getattr(torch.optim, cfg.OPT)

        if cfg.OPT == 'Adam':
            print("Using Adam >>> learning rate = {:4.3e}, momentum = {:4.3e}, weight decay = {:4.3e}".format(cfg.LR, cfg.MOMENTUM, cfg.WEIGHT_DECAY))
            upd = torch.optim.Adam(params, lr=cfg.LR, \
                                   betas=(cfg.BETA1, 0.999), \
                                   weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPT == 'SGD':
            print("Using SGD >>> learning rate = {:4.3e}, momentum = {:4.3e}, weight decay = {:4.3e}".format(cfg.LR, cfg.MOMENTUM, cfg.WEIGHT_DECAY))
            upd = torch.optim.SGD(params, lr=cfg.LR, \
                                  momentum=cfg.MOMENTUM, \
                                  nesterov=cfg.OPT_NESTEROV, \
                                  weight_decay=cfg.WEIGHT_DECAY)

        else:
            upd = optim(params, lr=cfg.LR)

        upd.zero_grad()

        return upd

    @staticmethod
    def set_lr(optim, lr):
        for param_group in optim.param_groups:
            param_group['lr'] = lr

    def _downsize(self, x, mode="bilinear"):
        x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(1)

        scale = min(*self.cfg.TB.IM_SIZE) / min(x.shape[-1], x.shape[-2])
        if mode == "nearest":
            x = F.interpolate(x, scale_factor=scale, mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=scale, mode=mode, align_corners=True)

        return x.squeeze(1)

    def _visualise_seg(self, epoch, outs, writer, tag, S = 5):

        def with_frame(image, mask, alpha=0.3):
            return alpha * image + (1 - alpha) * mask

        frames = outs["frames"][::S]
        frames_norm = self.denorm(frames.cpu().clone())
        frames_down = self._downsize(frames_norm)
        T,C,h,w = frames_down.shape

        visuals = []
        visuals.append(frames_down)

        if "masks_gt" in outs:
            mask_rgb_gt = self._apply_cmap(outs["masks_gt"][::S].cpu(), palette_davis, rand=False)
            mask_rgb_gt = self._downsize(mask_rgb_gt)
            mask_rgb_gt = with_frame(frames_down, mask_rgb_gt)
            visuals.append(mask_rgb_gt)

        mask_rgb_idx = self._apply_cmap(outs["masks_pred_idx"][::S].cpu(), palette_davis, rand=False)
        mask_rgb_idx = self._downsize(mask_rgb_idx)
        mask_rgb_idx = with_frame(frames_down, mask_rgb_idx)
        visuals.append(mask_rgb_idx)

        #if "masks_pred_conf" in outs:
        conf = self._downsize(outs["masks_pred_conf"][::S].cpu())
        conf_rgb = self._error_rgb(conf, cm.get_cmap("plasma"), frames_down, 0.3)
        visuals.append(conf_rgb)

        visuals = [x.float() for x in visuals]
        visuals = torch.cat(visuals, -1)

        self._visualise_grid(writer, visuals, epoch, tag)

    def _visualise(self, epoch, outs, T, writer, tag):
        visuals = []

        def overlay(mask, image, alpha=0.3):
            return alpha * image + (1 - alpha) * mask

        frames_orig = outs["frames_orig"]
        frames_orig = self.denorm(frames_orig.cpu().clone())
        frames_orig = self._downsize(frames_orig)
        visuals.append(frames_orig)

        frames = outs["frames"]
        frames_norm = self.denorm(frames.cpu().clone())
        frames_down = self._downsize(frames_norm)

        if "grid_mask" in outs:
            val_mask = outs["grid_mask"]
            val_mask = self._downsize(val_mask)
            val_mask = val_mask.unsqueeze(1).expand(-1,3,-1,-1).cpu()

            val_mask = overlay(val_mask, frames_orig)
            visuals.append(val_mask)

        if "map_target" in outs:
            val = outs["map_target"]
            val = self._apply_cmap(val)
            val = self._downsize(val, "nearest")
            val = overlay(val, frames_down)
            visuals.append(val)

        if "map_soft" in outs:
            val = outs["map_soft"]
            val = self._mask_rgb(val)
            val = self._downsize(val)
            visuals.append(val)

        visuals.append(frames_down)

        frames2 = outs["frames2"]
        frames2_norm = self.denorm(frames2.cpu().clone())
        frames2_down = self._downsize(frames2_norm)
        visuals.append(frames2_down)

        if "map" in outs:
            val = outs["map"]
            val = self._apply_cmap(val)
            val = self._downsize(val, "nearest").cpu()
            val = overlay(val, frames2_down)
            visuals.append(val)

        if "map_target_soft" in outs:
            val = outs["map_target_soft"]
            val = self._mask_rgb(val)
            val = self._downsize(val)
            visuals.append(val)

        # embedding error mask
        if "error_map" in outs:
            err_mask = outs["error_map"]
            err_mask = (err_mask - err_mask.min()) / (err_mask.max() - err_mask.min() + 1e-8)
            err_mask_rgb = self._error_rgb(err_mask, cmap=cm.get_cmap("plasma"), alpha=0.5)
            err_mask_rgb = self._downsize(err_mask_rgb)
            visuals.append(err_mask_rgb)

        if "aff_mask1" in outs:
            aff_mask = outs["aff_mask1"].unsqueeze(1).expand(-1,3,-1,-1).cpu()
            aff_mask = self._downsize(aff_mask)

            aff_frames = frames_orig.clone()
            aff_frames[::T] = overlay(aff_mask, aff_frames[::T], 0.5)
            visuals.append(aff_frames)

            aff_mask1 = self._error_rgb(outs["aff11"], cm.get_cmap("inferno"))
            aff_mask1 = self._downsize(aff_mask1)
            aff_mask1 = overlay(aff_mask1, frames_orig, 0.3)
            visuals.append(aff_mask1)

            aff_mask2 = self._error_rgb(outs["aff12"], cm.get_cmap("inferno"))
            aff_mask2 = self._downsize(aff_mask2)
            aff_mask2 = overlay(aff_mask2, frames2_down, 0.3)
            visuals.append(aff_mask2)

        if "aff_mask2" in outs:
            aff_mask = outs["aff_mask2"].unsqueeze(1).expand(-1,3,-1,-1).cpu()
            aff_mask = self._downsize(aff_mask)

            aff_frames = frames_down.clone()
            aff_frames[::T] = overlay(aff_mask, aff_frames[::T], 0.5)
            visuals.append(aff_frames)

            aff_mask1 = self._error_rgb(outs["aff21"], cm.get_cmap("inferno"))
            aff_mask1 = self._downsize(aff_mask1)
            aff_mask1 = overlay(aff_mask1, frames_orig, 0.3)
            visuals.append(aff_mask1)

            aff_mask2 = self._error_rgb(outs["aff22"], cm.get_cmap("inferno"))
            aff_mask2 = self._downsize(aff_mask2)
            aff_mask2 = overlay(aff_mask2, frames2_down, 0.3)
            visuals.append(aff_mask2)

        visuals = [x.cpu().float() for x in visuals]
        visuals = torch.cat(visuals, -1)

        self._visualise_grid(writer, visuals, epoch, tag, 4 * T)

    def save_vis_batch(self, key, batch):

        if self.vis_batch is None:
            self.vis_batch = {}

        if key in self.vis_batch:
            return

        batch_items = []
        for el in batch:
            el = el.clone().cpu() if torch.is_tensor(el) else el
            batch_items.append(el)

        self.vis_batch[key] = batch_items

    def has_vis_batch(self, key):
        return (not self.vis_batch is None and \
                    key in self.vis_batch)

    def _mask_rgb(self, masks, image_norm=None, palette=None, alpha=0.3):

        if palette is None:
            palette = self.loader.dataset.palette

        # visualising masks
        masks_conf, masks_idx = torch.max(masks, 1)
        masks_conf = masks_conf - F.relu(masks_conf - 1, 0)

        masks_idx_rgb = self._apply_cmap(masks_idx.cpu(), palette, mask_conf=masks_conf.cpu())
        if not image_norm is None:
            return alpha * image_norm + (1 - alpha) * masks_idx_rgb

        return masks_idx_rgb

    def _apply_cmap(self, mask_idx, palette=None, mask_conf=None, rand=True):

        if palette is None:
            palette = self.loader.dataset.palette

        ignore_mask = (mask_idx == -1).cpu()

        # cycle
        if rand:
            memsize = self.cfg.TRAIN.BATCH_SIZE * self.cfg.MODEL.GRID_SIZE**2
            mask_idx = ((mask_idx + 1) * 123) % memsize

        # convert mask to RGB
        mask = mask_idx.cpu().numpy().astype(np.uint32)
        mask_rgb = palette(mask)
        mask_rgb = torch.from_numpy(mask_rgb[:,:,:,:3])
        mask_rgb[ignore_mask] *= 0
        mask_rgb = mask_rgb.permute(0,3,1,2)

        if not mask_conf is None:
            # entropy
            mask_rgb *= mask_conf[:, None, :, :]

        return mask_rgb

    def _error_rgb(self, error_mask, cmap = cm.get_cmap('jet'), image=None, alpha=0.3):
        error_np = error_mask.cpu().numpy()

        # remove alpha channel
        error_rgb = cmap(error_np)[:, :, :, :3]
        error_rgb = np.transpose(error_rgb, (0,3,1,2))
        error_rgb = torch.from_numpy(error_rgb)

        if not image is None:
            return alpha * image + (1 - alpha) * error_rgb

        return error_rgb

    def _visualise_grid(self, writer, x_all, t, tag, T=1):
        
        # adding the labels to images
        bs, ch, h, w = x_all.size()
        x_all_new = torch.zeros(T, ch, h, w)
        for b in range(bs):

            x_all_new[b % T] = x_all[b]

            if (b + 1) % T == 0:
                summary_grid = vutils.make_grid(x_all_new, nrow=1, padding=8, pad_value=0.9).numpy()
                writer.add_image(tag + "_{:02d}".format(b // T), summary_grid, t)
                x_all_new.zero_()

    def visualise_results(self, epoch, writer, tag, step_func):
        # visualising
        self.net.eval()

        with torch.no_grad():
            step_func(epoch, self.vis_batch[tag], \
                      train=False, visualise=True, \
                      writer=writer, tag=tag)
