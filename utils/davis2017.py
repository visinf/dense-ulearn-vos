"""
Credit: https://github.com/davisvideochallenge/davis2017-evaluation.git
License: BSD 3-Clause
Copyright (c) 2020, DAVIS: Densely Annotated VIdeo Segmentation
"""

import sys
import numpy as np
from utils.davis2017_metrics import db_eval_boundary, db_eval_iou
import utils.davis2017_utils as utils

def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
    if all_res_masks.shape[0] > all_gt_masks.shape[0]:
        sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        sys.stdout.write("\nThe number of predictions is less than ground truth. Padding with zero.")
        zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    for ii in range(all_gt_masks.shape[0]):

        if 'J' in metric:
            j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            sys.stdout.flush()

        if 'F' in metric:
            f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            sys.stdout.flush()

    return j_metrics_res, f_metrics_res


def evaluate_semi(all_gt_masks, all_res_masks, metric=('J', 'F'), debug=False):
    metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
    if 'T' in metric:
        raise ValueError('Temporal metric not supported!')
    if 'J' not in metric and 'F' not in metric:
        raise ValueError('Metric possible values are J for IoU or F for Boundary')

    # Containers
    metrics_res = {}
    if 'J' in metric:
        metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    if 'F' in metric:
        metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    for seq, (seq_gt_masks, seq_res_masks) in enumerate(zip(all_gt_masks, all_res_masks)):

        seq_gt_masks, seq_res_masks = seq_gt_masks[:, 1:-1, :, :], seq_res_masks[:, 1:-1, :, :]
        j_metrics_res, f_metrics_res = _evaluate_semisupervised(seq_gt_masks, seq_res_masks, None, metric)

        for ii in range(seq_gt_masks.shape[0]):
            seq_name = f'{seq}_{ii+1}'
            if 'J' in metric:
                [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                metrics_res['J']["M"].append(JM)
                metrics_res['J']["R"].append(JR)
                metrics_res['J']["D"].append(JD)
                metrics_res['J']["M_per_object"][seq_name] = JM
            if 'F' in metric:
                [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                metrics_res['F']["M"].append(FM)
                metrics_res['F']["R"].append(FR)
                metrics_res['F']["D"].append(FD)
                metrics_res['F']["M_per_object"][seq_name] = FM

        # Show progress
        if debug:
            sys.stdout.write(seq + '\n')
            sys.stdout.flush()

    return metrics_res
