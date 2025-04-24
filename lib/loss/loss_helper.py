##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils.tools.logger import Logger as Log
import torch.distributed as dist

import einops

def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label
class FocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha = 0.004, size_average=True, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.reduction = reduction
    def forward_single(self, pred, true):
        """
        Compute the focal loss.

        Args:
        - pred (torch.Tensor): The predicted tensor with shape Bx1xH x W.
        - true (torch.Tensor): The ground truth tensor with shape Bx H x W.

        Returns:
        - loss (torch.Tensor): The computed Focal Loss.

        true = true.view(-1,1)
        if not (pred.size() == true.size()):
        """

        true = true.view(-1,1)
        if not (pred.size() == true.size()):
            raise ValueError("Input and target must have the same size")

        # Ensure the input tensors are of float type
        pred = pred.float()
        true = true.float()

        # Calculate the binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, true, reduction='none')

        # Calculate the probabilities
        probas = torch.sigmoid(pred)
        true = true.type(probas.dtype)

        # Calculate the focal loss components
        focal_loss = torch.where(true >= 0.5, (1. - probas) ** self.gamma * bce_loss,
                                 probas ** self.gamma * bce_loss)

        if self.alpha is not None:
            alpha_t = torch.ones_like(probas) * self.alpha
            alpha_t = torch.where(true >= 0.5, alpha_t, 1. - alpha_t)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def forward(self, input, target):
        if input.dim() > 2:
            if input.dim()<4:
                input = input.unsqueeze(dim=0)
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))

        num_classes = input.shape[1]
        if num_classes == 1:
           return self.forward_single(input, target)



        target = target.view(-1, 1)
        tmp_target = target.clone()
        tmp_target[tmp_target == -1] = 0
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, tmp_target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean',ignore_idx = -1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.ignore_val  = -1
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, predict, target, valid_mask = None,**kwargs):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"


        predict = predict.contiguous().view(predict.shape[0], -1)
        #target = torch.where(target > 0, torch.ones_like(target), target)  # 将有效标签转换为1
        target = target.contiguous().view(target.shape[0], -1)
        mask = target != self.ignore_val
        if valid_mask is not None:
            valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1) *mask
            num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + self.smooth
            den = torch.sum((predict.pow(self.p) + target.pow(self.p)) * valid_mask, dim=1) + self.smooth
        else:
            num = torch.sum(torch.mul(predict, target), dim=1) * 2 + self.smooth
            den = torch.sum((predict.pow(self.p) + target.pow(self.p)), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    """Computes the Sørensen–Dice loss.
                   Note that PyTorch optimizers minimize a loss. In this
                   case, we would like to maximize the dice loss so we
                   return the negated dice loss.
                   Args:
                       true: a tensor of shape [B, 1, H, W].
                       logits: a tensor of shape [B, C, H, W]. Corresponds to
                           the raw output or logits of the model.
                       eps: added to the denominator for numerical stability.
                   Returns:
                       dice_loss: the Sørensen–Dice loss.
                   """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def dice_loss_logits(self, pred, true, epsilon=1e-6):
        """
        Compute the Dice Loss.

        Args:
        - pred (torch.Tensor): The predicted tensor with shape BX1X512X512.
        - true (torch.Tensor): The ground truth tensor with shape BX1X512X512.
        - epsilon (float): A small value to avoid division by zero.

        Returns:
        - dice_loss (torch.Tensor): The computed Dice Loss.
        """

        # Flatten the tensors to make the computation easier
        pred_flat = pred.view(pred.size(0), -1)
        true_flat = true.view(true.size(0), -1)

        # Compute the intersection
        intersection = (pred_flat * true_flat).sum(dim=1)

        # Compute the union
        union = pred_flat.sum(dim=1) + true_flat.sum(dim=1)

        # Compute the Dice coefficient (score)
        dice_score = (2. * intersection + epsilon) / (union + epsilon)

        # Compute the Dice loss
        dice_loss = 1 - dice_score

        # Return the mean Dice loss over the batch
        return dice_loss.mean()

    # def forward(self, logits, true, eps=1e-7):
    #     """
    #     logits: [B, num_classes, H, W]
    #     true: [B, 1, H, W], 值为0 ~ num_classes-1
    #     """
    #     if logits.dim() < 4:
    #         logits = logits.unsqueeze(dim=0)
    #
    #     num_classes = logits.shape[1]
    #
    #     if num_classes == 1:
    #         # 二分类 (binary)
    #         probs = torch.sigmoid(logits)  # [B,1,H,W]
    #         true = true.float()
    #     else:
    #         # 多分类 (multi-class)
    #         probs = F.softmax(logits, dim=1)  # [B,C,H,W]
    #
    #         # 将标签转为one-hot编码 [B,C,H,W]
    #         true = F.one_hot(true.squeeze(1).long(), num_classes)  # [B,H,W,C]
    #         true = true.permute(0, 3, 1, 2).float()  # [B,C,H,W]
    #
    #     # 展开到二维
    #     probs_flat = probs.reshape(probs.shape[0], probs.shape[1], -1)
    #     true_flat = true.reshape(true.shape[0], true.shape[1], -1)
    #
    #     intersection = (probs_flat * true_flat).sum(-1)
    #     cardinality = probs_flat.sum(-1) + true_flat.sum(-1)
    #
    #     dice_loss = (2 * intersection + eps) / (cardinality + eps)
    #     dice_loss = 1 - dice_loss.mean()
    #
    #     return dice_loss

    def forward(self,logits,true,eps = 1e-7):

        if logits.dim() < 4:
            logits = logits.unsqueeze(dim=0)
        num_classes = logits.shape[1]
        #tmp_target = target.clone()
        true[true == -1] = 0
        if num_classes == 1:
            pos_prob = torch.sigmoid(logits)
            return self.dice_loss_logits(pos_prob, true)
            # true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            # true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            # true_1_hot_f = true_1_hot[:, 0:1, :, :]
            # true_1_hot_s = true_1_hot[:, 1:2, :, :]
            # true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            # pos_prob = torch.sigmoid(logits)
            # neg_prob = 1 - pos_prob
            # probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes).to(true.device)

            true_1_hot = true_1_hot[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)
class hybrid_loss(nn.Module):
    def __init__(self, configer, reduction = 'mean', dice_w = 1.0,from_model = True):
        super(hybrid_loss, self).__init__()
        self.configer = configer

        self.focal_loss = FocalLoss()
        self.w_dice = dice_w

        self.dice_loss = DiceLoss()
        self.from_model = from_model



    def forward(self,preds, targets,**kwargs):
        # loss = torch.FloatTensor(0.0)
        # return  {"loss":loss,"preds":preds}

        h, w = targets.shape[-2:]
        preds = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True).squeeze(1)
        if isinstance(preds,tuple) or isinstance(preds,list):
            loss = 0.0
            for pred in preds:
                target = F.interpolate(targets.float().unsqueeze(1),size = pred.shape[-2:],mode='nearest').squeeze(1).long()
                bce = self.focal_loss(pred, target)
                dice = self.dice_loss(pred, target)
                loss += bce + dice
            return {"loss":loss,"preds":preds[-1]}

        loss = 0.0
        focal_loss = self.focal_loss(preds, targets)
        dice = self.dice_loss(preds, targets)
        loss = focal_loss  + dice *  self.w_dice
        return {"loss":loss}

    # def forward(self,model, inputs_left,inputs_right, targets,**kwargs):
    #     preds = model(inputs_left,inputs_right)
    #
    #
    #     if isinstance(preds,tuple) or isinstance(preds,list):
    #         loss = 0.0
    #         for pred in preds:
    #             target = F.interpolate(targets.float().unsqueeze(1),size = pred.shape[-2:],mode='nearest').squeeze(1).long()
    #             bce = self.focal_loss(pred, target)
    #             dice = self.dice_loss(pred, target)
    #             loss += bce + dice
    #         return {"loss":loss,"preds":preds[0]}
    #
    #     loss = 0.0
    #     bce = self.focal_loss(preds, targets)
    #     dice = self.dice_loss(preds, targets)
    #     loss += bce + dice
    #     return {"loss":loss,"preds":preds}
from abc import ABC
class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(PixelContrastLoss, self).__init__()

        self.configer = configer
        self.temperature = self.configer.get('contrast', 'temperature')
        self.base_temperature = self.configer.get('contrast', 'base_temperature')

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

        self.max_samples = self.configer.get('contrast', 'max_samples')
        self.max_views = self.configer.get('contrast', 'max_views')
        self.class_flags = self.configer.get('contrast', 'class_flags')

    def _high_entropy_anchor_sampling(self,feats,label,logits,mask = None):
        '''
            params:
            feats: pixel embedding features,dim * (B * H * W)
            label: gt of per pixel,(B * H * W)
            logits:output of net
        '''
        batch_size, feat_dim = feats.shape[0], feats.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = label[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if torch.sum(this_y == x) > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = logits[ii]
            this_y = label[ii]
            this_classes = classes[ii]
            this_feats = feats[ii]
            for cls_id in this_classes:
                p = this_y_hat[cls_id,:]
                mask = this_y == cls_id
                plogp = -1.0 * p * torch.log(p)
                plogp = plogp[mask]
                mask_feats = this_feats[mask]
                num_hard = int(n_view / 2)
                num_easy = n_view - num_hard

                #sort = torch.sort()

                _,hard_indices = torch.topk(plogp,num_hard,dim=0,)
                _,easy_indices = torch.topk(plogp,num_easy,largest=False,dim=0)

                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = mask_feats[indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_




    def _build_anchor_sampling(self, X, y_hat,y):

        if len(y_hat.shape) == 3:
            y_hat = torch.argmax(y_hat, dim=1)

        if y.shape[2:] != y_hat.shape[2:]:
            b, h, w = y_hat.shape
            y = F.interpolate(y, (h, w), mode='nearest')

        batch_size, feat_dim = X.shape[0],X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if torch.sum(this_y == x) > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)
        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)



        #
        # X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        # y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        x_anchors = []
        y_anchors = []
       # X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            for cls_id in this_classes:
                if cls_id in self.class_flags:
                    hard_indices = ((this_y == cls_id) & (this_y_hat != cls_id)).nonzero(as_tuple=False)
                    easy_indices = ((this_y == cls_id) & (this_y_hat == cls_id)).nonzero(as_tuple=False)
                else:
                    easy_indices = ((this_y == cls_id) & (this_y_hat == cls_id)).nonzero(as_tuple=False)
                    hard_indices = easy_indices

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_hard_keep = min(n_view, num_hard)
                    num_easy_keep = n_view - num_hard_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    continue
                    #raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                x_anchors.append(X[ii, indices, :].squeeze(1))
                y_anchors.append(cls_id)
                #X_ptr += 1

        X_ = torch.stack(x_anchors,dim=0)
        y_ = torch.stack(y_anchors, dim=0)
        return X_, y_
    def _hard_anchor_sampling(self, X, y_hat, y, mask = None):





        # if len(X.shape) == 3:
        #     b,m,c = X.shape
        #     X = einops.rearrange(X,'b (m1 m2) c -> b m1 m2 c', m1 = int(math.sqrt(m)))
        if len(y_hat.shape) == 3:
           y_hat = torch.argmax(y_hat,dim=1)

        if y.shape[2:] != y_hat.shape[2:]:
            b,h,w = y_hat.shape
            y = F.interpolate(y,(h,w),mode='nearest')

        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if torch.sum(this_y == x) > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:




                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero(as_tuple = False)
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero(as_tuple = False)

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            #if ii == 0: continue
            this_q = Q[ii, :cache_size, :]

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _contrastive_self(self, x_anchor, y_anchor):
        """
        计算当前样本与其相同类别样本之间的对比损失。
        x_anchor: 当前样本特征，形状为 [B, n_view, dim]
        y_anchor: 当前样本的类别标签，形状为 [B]
        """
        batch_size, n_view, feat_dim = x_anchor.shape  # 获取 batch 大小、视图数量和特征维度

        # 将 x_anchor 特征展平（形状：[B, n_view, dim] -> [B*n_view, dim]）
        anchor_feature = x_anchor.view(batch_size * n_view, feat_dim)  # 展平为 [B*n_view, dim]
        anchor_feature = nn.functional.normalize(anchor_feature, p=2, dim=1)  # 对特征进行归一化

        # 将 y_anchor 转换为一维向量
        y_anchor = y_anchor.view(batch_size)  # [B]

        # 创建一个正负样本的 mask，正样本为同类别的样本，负样本为不同类别的样本
        # 对于同一批次中的样本，我们需要检查它们的类别是否相同
        mask = torch.eq(y_anchor.unsqueeze(1), y_anchor.unsqueeze(0))  # [B, B]，同类别为 1，不同类别为 0
        mask = mask.unsqueeze(1).repeat(1, n_view, 1).repeat(1, 1, n_view).view(batch_size * n_view,batch_size * n_view).float()

        # 计算当前样本与自身类别的对比损失
        anchor_dot_self = torch.matmul(anchor_feature, anchor_feature.T)  # [B*n_view, B*n_view]

        # 获取每个样本的最大相似度（对角线元素）
        self_logits_max, _ = torch.max(anchor_dot_self, dim=1, keepdim=True)
        self_logits = anchor_dot_self - self_logits_max.detach()  # 减去最大值以稳定训练

        # 计算正样本和负样本的 exp(logits)
        exp_self_logits = torch.exp(self_logits)

        # 使用 mask 来选择正样本和负样本
        exp_pos = (exp_self_logits * mask).sum(dim=1, keepdim=True)  # 同类别样本的相似度
        exp_neg = (exp_self_logits * (1 - mask)).sum(dim=1, keepdim=True)  # 不同类别样本的相似度

        # 自身对比损失
        self_contrastive_loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-8) + 1e-8).mean()

        return self_contrastive_loss

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)
        anchor_feature = nn.functional.normalize(anchor_feature, p=2, dim=1)
        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        mask = torch.eq(y_anchor, y_contrast.T).float().to(anchor_feature.device)

        # 50 x 8000


        # contrast_feature = nn.functional.normalize(contrast_feature, p=2, dim=1)
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)

        # 正样本 (去除自身)
        pos_mask = mask * logits_mask

        # 负样本
        neg_mask = (1 - mask)

        # exp(logits)
        exp_logits = torch.exp(logits) * logits_mask

        # 正负样本exp和分别计算：
        exp_pos = (exp_logits * pos_mask).sum(dim=1, keepdim=True)  # 正样本
        exp_neg = (exp_logits * neg_mask).sum(dim=1, keepdim=True)  # 负样本

        loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-8) + 1e-8).mean()




        # Loss 计算
        #loss = -(self.base_temperature / self.temperature) * loss
        return  loss
    def _contrastive_new(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        #anchor_count = n_view
        anchor_count = n_view * anchor_num
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        #mask = torch.eq(y_anchor, y_contrast.T).float().cuda()


        logits = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature

        # 构建正负样本mask
        mask = torch.eq(y_anchor, y_contrast.T).float().to(anchor_feature.device)
        logits_mask = 1 - torch.eye(mask.shape[0], device=mask.device)
        pos_mask = mask * logits_mask  # 正样本mask (去掉自己)
        neg_mask = (1 - mask)  # 负样本mask

        # 计算exp(logits)
        exp_logits = torch.exp(logits) * logits_mask  # 去掉对角线影响

        # 计算分母 (正样本 + 负样本)
        exp_pos = (exp_logits * pos_mask).sum(dim=1)  # [B*n_view]
        exp_neg = (exp_logits * neg_mask).sum(dim=1)  # [B*n_view]

        # 计算loss (标准InfoNCE)
        loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-6) + 1e-6).mean()

        return loss
        # anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
        #                                 self.temperature)
        #
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        #
        # #logits_mask = torch.ones_like(mask).scatter_(1,torch.arange(anchor_count).view(-1,1).cuda(),0)
        # #mask = mask.repeat(anchor_count, contrast_count)
        # #mask = mask * logits_mask
        # neg_mask = 1 - mask
        #
        # exp_logits = torch.exp(logits)
        #
        # pos_feats = exp_logits[mask.bool()]
        # neg_feats = exp_logits[neg_mask.bool()]
        # loss = -1.0 * torch.mean(torch.log(pos_feats / (pos_feats.sum() + neg_feats.sum() + 1e-6) + 1e-6))
        # return loss



    def forward(self, feats, predict=None,  labels=None, queue=None,mask = None):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        preds = torch.nn.functional.interpolate(predict,
                                                 (feats.shape[2], feats.shape[3]), mode='bilinear',align_corners=True)
        b,c,_,_ = preds.shape
        preds = torch.softmax(preds,dim=1)
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        #batch_size = feats.shape[0]

        labels = labels.contiguous().view(b, -1)
        preds = preds.contiguous().view(b,c, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        feats_, labels_ = self._build_anchor_sampling(feats, preds, labels)
        #feats_, labels_ = self._build_anchor_sampling(feats, preds, labels,mask)
        #feats_, labels_ = self._high_entropy_anchor_sampling(feats, labels, preds, mask)
        if feats_ is None or labels_ is None:
            return  0.0
        #loss_self = self._contrastive_self(feats_, labels_)
        loss_memory = self._contrastive(feats_, labels_, queue=queue)

        loss = loss_memory# + loss_self
        return loss
class Contrast_loss(nn.Module):
    def __init__(self, configer,reduction = 'mean',from_model = True):
        super(Contrast_loss, self).__init__()
        self.configer = configer
        self.ce_loss = hybrid_loss(configer) # 交叉熵Loss（Focal loss + Dice Loss）

        self.infoNCE_Loss = PixelContrastLoss(configer)
        self.from_model = from_model

    @torch.no_grad()
    def __init_queue(self, keys, target, queue, mem_size = 2500, n = 200):
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]
        labels = F.interpolate(target.float(), size=keys.shape[2:], mode="nearest").long()
        ptr = torch.zeros(labels.max() + 1, dtype=torch.long, device=keys.device)
        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)

            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x >= 0]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero(as_tuple=False)

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, n)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)

                if ptr + K >= mem_size:
                    queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    ptr[lb] = 0
                else:
                    queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    ptr = (ptr[lb] + 1) % mem_size

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, pixel_queue, pixel_queue_ptr):

        mem_size = pixel_queue.shape[1]
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]
        labels = F.interpolate(labels.float().unsqueeze(1), size=keys.shape[2:], mode="nearest").long()

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)

            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x >= 0]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero(as_tuple=False)

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, 100)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(pixel_queue_ptr[lb].item())

                if ptr + K >= mem_size:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % mem_size

    def forward(self, model, preds, targets, with_contrast = False, **kwargs):
        # loss = torch.FloatTensor(0.0)
        # return  {"loss":loss,"preds":preds}
        assert "dense_embed" in preds.keys()
        assert "pred" in preds.keys()
        preds_dense = preds["pred"]
        preds_corse = preds["pred_aux"]
        embeddings = preds["dense_embed"]

        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        queue = model.mem_queue
        queue_ptr = model.mem_queue_ptr

        dense_loss = self.ce_loss(preds_dense, targets)["loss"]
        corse_loss = self.ce_loss(preds_corse,targets)["loss"]
        # if isinstance(ce_loss, dict):
        #     ce_loss = ce_loss["loss"]
        ce_loss = dense_loss + 0.3* corse_loss
        if embeddings is not None:
            if with_contrast:
                info_loss = self.infoNCE_Loss(embeddings, preds_dense, targets, queue = queue)
                self._dequeue_and_enqueue(embeddings,targets,queue,queue_ptr)
            else:
                if self.training:
                    self._dequeue_and_enqueue(embeddings, targets, queue, queue_ptr)
                    #print(queue_ptr.max())
                #self.__init_queue(embeddings,targets, queue)
                info_loss = ce_loss * 0.0

            loss = ce_loss + info_loss *0.1
        else:
            loss = ce_loss
        return {"loss":loss,"preds":preds_dense, "preds_aux":preds_corse}








