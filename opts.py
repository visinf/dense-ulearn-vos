"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

from __future__ import print_function

import os
import torch
import argparse
from core.config import cfg

def add_global_arguments(parser):

    #
    # Model details
    #
    parser.add_argument("--snapshot-dir", type=str, default='./snapshots',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="Where to save log files of the model.")
    parser.add_argument("--exp", type=str, default="main",
                        help="ID of the experiment (multiple runs)")
    parser.add_argument("--run", type=str, help="ID of the run")
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--seed', default=64, type=int, help='seed for initializing training. ')

    # 
    # Inference only
    #
    parser.add_argument("--infer-list", default="voc12/val.txt", type=str)
    parser.add_argument('--mask-output-dir', type=str, default=None, help='path where to save masks')
    parser.add_argument("--resume", type=str, default=None, help="Snapshot \"ID,iter\" to load")

    #
    # Configuration
    #
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_global_arguments(args):

    args.cuda = torch.cuda.is_available()
    print("Available threads: ", torch.get_num_threads())

    args.logdir = os.path.join(args.logdir, args.exp, args.run)
    maybe_create_dir(args.logdir)

    #
    # Model directories
    #
    args.snapshot_dir = os.path.join(args.snapshot_dir, args.exp, args.run)
    maybe_create_dir(args.snapshot_dir)

def get_arguments(args_in):
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Dense Unsupervised Learning for Video Segmentation")

    add_global_arguments(parser)
    args = parser.parse_args(args_in)
    check_global_arguments(args)

    return args
