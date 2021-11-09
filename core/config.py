from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import yaml
import six
import os
import os.path as osp
import copy
from ast import literal_eval

import numpy as np
from packaging import version

from utils.collections import AttrDict

__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.NUM_GPUS = 1
# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

# ---------------------------------------------------------------------------- #
# Training schedule options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()
__C.TRAIN.BATCH_SIZE = 20
__C.TRAIN.NUM_EPOCHS = 15
__C.TRAIN.NUM_WORKERS = 4
__C.TRAIN.TASK = "YTVOS"
__C.TRAIN.BLOCK_BN = False
__C.TRAIN.STOP_GRAD = True

# ---------------------------------------------------------------------------- #
# Dataset options (+ augmentation options)
# ---------------------------------------------------------------------------- #
__C.DATASET = AttrDict()

__C.DATASET.CROP_SIZE = [256,256]
__C.DATASET.SMALLEST_RANGE = [256,320]
__C.DATASET.RND_CROP  = False
__C.DATASET.RND_HFLIP = True
__C.DATASET.MP4 = False

__C.DATASET.RND_ZOOM = False
__C.DATASET.RND_ZOOM_RANGE = [.5,1.]
__C.DATASET.GUIDED_HFLIP = True

__C.DATASET.ROOT = "data"
__C.DATASET.VIDEO_LEN = 5
__C.DATASET.FRAME_GAP = 5
__C.DATASET.VAL_FRAME_GAP = 2

# size of the augmentation
# (how many augmented copies to create for 1 image)
__C.DATASET.NUM_CLASSES = 7

# inference-time parameters
__C.TEST = AttrDict()
__C.TEST.RADIUS = 12
__C.TEST.TEMP = 0.05
__C.TEST.KNN = 10
__C.TEST.CXT_SIZE = 20
__C.TEST.INPUT_SIZE = -1
__C.TEST.KEY = "res4"

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()
__C.MODEL.ARCH = 'resnet18'
__C.MODEL.LR_SCHEDULER = 'poly'
__C.MODEL.LR_STEP = 5
__C.MODEL.LR_GAMMA = 0.1 # divide by 10 every LR_STEP epochs
__C.MODEL.LR_POWER = 1.0
__C.MODEL.LR_SCHED_USE_EPOCH = True
__C.MODEL.OPT = 'Adam'
__C.MODEL.OPT_NESTEROV = False
__C.MODEL.LR = 3e-4
__C.MODEL.BETA1 = 0.5
__C.MODEL.MOMENTUM = 0.9
__C.MODEL.WEIGHT_DECAY = 1e-5
__C.MODEL.REMOVE_LAYERS = []
__C.MODEL.FEATURE_DIM = 128
__C.MODEL.GRID_SIZE = 8
__C.MODEL.GRID_SIZE_REF = 4
__C.MODEL.CE_REF = 0.1

# ---------------------------------------------------------------------------- #
# Options for refinement
# ---------------------------------------------------------------------------- #
__C.LOG = AttrDict()
__C.LOG.ITER_VAL = 1
__C.LOG.ITER_TRAIN = 10

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.TB = AttrDict()
__C.TB.IM_SIZE = (196, 196) # image resolution

# [Infered value]
__C.CUDA = False
__C.DEBUG = False

# [Infered value]
__C.PYTORCH_VERSION_LESS_THAN_040 = False

def assert_and_infer_cfg(make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if make_immutable:
        cfg.immutable(True)


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))
    _merge_a_into_b(yaml_cfg, __C)

cfg_from_file = merge_cfg_from_file


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value

cfg_from_list = merge_cfg_from_list


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
