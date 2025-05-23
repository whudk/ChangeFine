#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Repackage some mask operations.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pycocotools.mask as mask_util


class MaskHelper(object):

    @staticmethod
    def polys2mask(polygons, width, height):
        rles = mask_util.frPyObjects(polygons, height, width)
        rle = mask_util.merge(rles)
        mask = np.array(mask_util.decode(rle), dtype=np.float32)
        return mask

    @staticmethod
    def rle2mask(rle, width, height):
        if 'counts' in rle and type(rle['counts']) == list:
            rle = mask_util.frPyObjects([rle], height, width)

        mask = np.array(mask_util.decode(rle), dtype=np.float32)
        # Flatten in case polygons was a list
        mask = np.sum(mask, axis=2)
        mask = np.array(mask > 0, dtype=np.float32)
        return mask

    @staticmethod
    def polys2mask_wrt_box(polygons, box, target_size):
        """Convert from the COCO polygon segmentation format to a binary mask
        encoded as a 2D array of data type numpy.float32. The polygon segmentation
        is understood to be enclosed in the given box and rasterized to an M x M
        mask. The resulting mask is therefore of shape (M, M).
        """
        w = box[2] - box[0]
        h = box[3] - box[1]

        w = np.maximum(w, 1)
        h = np.maximum(h, 1)

        polygons_norm = []
        for poly in polygons:
            p = np.array(poly, dtype=np.float32)
            p[0::2] = (p[0::2] - box[0]) * target_size[0] / w
            p[1::2] = (p[1::2] - box[1]) * target_size[1] / h
            polygons_norm.append(p)

        rle = mask_util.frPyObjects(polygons_norm, target_size[1], target_size[0])
        mask = np.array(mask_util.decode(rle), dtype=np.float32)
        # Flatten in case polygons was a list
        mask = np.sum(mask, axis=2)
        mask = np.array(mask > 0, dtype=np.float32)
        return mask

    @staticmethod
    def rle_mask_voting(top_masks, all_masks, all_dets, iou_thresh, binarize_thresh, method='AVG'):
        """Returns new masks (in correspondence with `top_masks`) by combining
        multiple overlapping masks coming from the pool of `all_masks`. Two methods
        for combining masks are supported: 'AVG' uses a weighted average of
        overlapping mask pixels; 'UNION' takes the union of all mask pixels.
        """
        if len(top_masks) == 0:
            return

        all_not_crowd = [False] * len(all_masks)
        top_to_all_overlaps = mask_util.iou(top_masks, all_masks, all_not_crowd)
        decoded_all_masks = [
            np.array(mask_util.decode(rle), dtype=np.float32) for rle in all_masks
        ]
        decoded_top_masks = [
            np.array(mask_util.decode(rle), dtype=np.float32) for rle in top_masks
        ]
        all_boxes = all_dets[:, :4].astype(np.int32)
        all_scores = all_dets[:, 4]

        # Fill box support with weights
        mask_shape = decoded_all_masks[0].shape
        mask_weights = np.zeros((len(all_masks), mask_shape[0], mask_shape[1]))
        for k in range(len(all_masks)):
            ref_box = all_boxes[k]
            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, mask_shape[1])
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, mask_shape[0])
            mask_weights[k, y_0:y_1, x_0:x_1] = all_scores[k]
        mask_weights = np.maximum(mask_weights, 1e-5)

        top_segms_out = []
        for k in range(len(top_masks)):
            # Corner case of empty mask
            if decoded_top_masks[k].sum() == 0:
                top_segms_out.append(top_masks[k])
                continue

            inds_to_vote = np.where(top_to_all_overlaps[k] >= iou_thresh)[0]
            # Only matches itself
            if len(inds_to_vote) == 1:
                top_segms_out.append(top_masks[k])
                continue

            masks_to_vote = [decoded_all_masks[i] for i in inds_to_vote]
            if method == 'AVG':
                ws = mask_weights[inds_to_vote]
                soft_mask = np.average(masks_to_vote, axis=0, weights=ws)
                mask = np.array(soft_mask > binarize_thresh, dtype=np.uint8)
            elif method == 'UNION':
                # Any pixel that's on joins the mask
                soft_mask = np.sum(masks_to_vote, axis=0)
                mask = np.array(soft_mask > 1e-5, dtype=np.uint8)
            else:
                raise NotImplementedError('Method {} is unknown'.format(method))
            rle = mask_util.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
            top_segms_out.append(rle)

        return top_segms_out

    @staticmethod
    def rle_mask_nms(masks, dets, thresh, mode='IOU'):
        """Performs greedy non-maximum suppression based on an overlap measurement
        between masks. The type of measurement is determined by `mode` and can be
        either 'IOU' (standard intersection over union) or 'IOMA' (intersection over
        mininum area).
        """
        if len(masks) == 0:
            return []
        if len(masks) == 1:
            return [0]

        if mode == 'IOU':
            # Computes ious[m1, m2] = area(intersect(m1, m2)) / area(union(m1, m2))
            all_not_crowds = [False] * len(masks)
            ious = mask_util.iou(masks, masks, all_not_crowds)
        elif mode == 'IOMA':
            # Computes ious[m1, m2] = area(intersect(m1, m2)) / min(area(m1), area(m2))
            all_crowds = [True] * len(masks)
            # ious[m1, m2] = area(intersect(m1, m2)) / area(m2)
            ious = mask_util.iou(masks, masks, all_crowds)
            # ... = max(area(intersect(m1, m2)) / area(m2),
            #           area(intersect(m2, m1)) / area(m1))
            ious = np.maximum(ious, ious.transpose())
        elif mode == 'CONTAINMENT':
            # Computes ious[m1, m2] = area(intersect(m1, m2)) / area(m2)
            # Which measures how much m2 is contained inside m1
            all_crowds = [True] * len(masks)
            ious = mask_util.iou(masks, masks, all_crowds)
        else:
            raise NotImplementedError('Mode {} is unknown'.format(mode))

        scores = dets[:, 4]
        order = np.argsort(-scores)

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            ovr = ious[i, order[1:]]
            inds_to_keep = np.where(ovr <= thresh)[0]
            order = order[inds_to_keep + 1]

        return keep