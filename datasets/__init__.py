"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import torch
from torch.utils import data

from .dataloader_seg import DataSeg
from .dataloader_video import DataVideo, DataVideoKinetics

def get_sets(task):

    # fetch the names of data lists depending on the task
    sets = {}
    if task == "OxUvA":
        sets["train_video"] = "train_oxuva"
        sets["val_video"] = "val_davis2017_480p"
        sets["val_video_seg"] = "val2_davis2017_480p"
    elif task == "YTVOS":
        sets["train_video"] = "train_ytvos"
        sets["val_video"] = "val_davis2017_480p"
        sets["val_video_seg"] = "val2_davis2017_480p"
    elif task == "TrackingNet":
        sets["train_video"] = "train_tracking"
        sets["val_video"] = "val_davis2017_480p"
        sets["val_video_seg"] = "val2_davis2017_480p"
    elif task == "kinetics400":
        sets["train_video"] = "train_kinetics400"
        sets["val_video"] = "val_davis2017_480p"
        sets["val_video_seg"] = "val2_davis2017_480p"
    else:
        raise NotImplementedError("Dataset '{}' not recognised.".format(task))
    
    return sets

def get_dataloader(args, cfg, split):
    assert split in ("train", "val")

    task = cfg.TRAIN.TASK
    data_sets = get_sets(cfg.TRAIN.TASK)

    # total batch size: # of GPUs * batch size per GPU
    batch_size = cfg.TRAIN.BATCH_SIZE
    kwargs = {'pin_memory': True, 'num_workers': args.workers}
    print("Dataloader: # workers {}".format(args.workers))

    def _dataloader(dataset, batch_size, shuffle=True, drop_last=False):
        return data.DataLoader(dataset, batch_size=batch_size, \
                               shuffle=shuffle, drop_last=drop_last, **kwargs)


    print("Split: ", split)
    if split == "train":
        VideoLoader = DataVideoKinetics if cfg.DATASET.MP4 else DataVideo
        dataset_video = VideoLoader(cfg, data_sets["train_video"])
        return _dataloader(dataset_video, batch_size, drop_last=True)
    else:
        dataset_video = DataVideo(cfg, data_sets["val_video"], val=True)
        dataset_video_seg = DataSeg(cfg, data_sets["val_video_seg"])

        return {"val_video": _dataloader(dataset_video, batch_size, shuffle=False), \
                "val_video_seg": _dataloader(dataset_video_seg, 1, shuffle=False)}
