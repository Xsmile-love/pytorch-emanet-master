import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc

class AnchorTargetCreator(object):
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample       = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio      = pos_ratio

    def __call__(self, bbox, anchor):
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):
        #----------------------------------------------#
        #   anchor and bbox of iou
        #   The obtained shape of ious is [num_anchors, num_gt]
        #----------------------------------------------#
        ious = bbox_iou(anchor, bbox)

        if len(bbox)==0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        #---------------------------------------------------------#
        #   Get the most corresponding real box for each a priori box [num_anchors, ]
        #---------------------------------------------------------#
        argmax_ious = ious.argmax(axis=1)
        #---------------------------------------------------------#
        #   Find the iou of the most corresponding real box for each a priori box [num_anchors, ]
        #---------------------------------------------------------#
        max_ious = np.max(ious, axis=1)
        #---------------------------------------------------------#
        #   Get the most corresponding a priori box for each real box [num_gt, ]
        #---------------------------------------------------------#
        gt_argmax_ious = ious.argmax(axis=0)
        #---------------------------------------------------------#
        #   Ensure the existence of a corresponding a priori box for each real box
        #---------------------------------------------------------#
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious
        
    def _create_label(self, anchor, bbox):
        # ------------------------------------------ #
        #   1 is a positive sample, 0 is a negative sample, -1 is ignored
        #   All set to -1 at initialization
        # ------------------------------------------ #
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        # ------------------------------------------------------------------------ #
        #   argmax_ious is the serial number of the largest real box corresponding to each a priori box         [num_anchors, ]
        #   max_ious is the iou of the largest real box corresponding to each real box             [num_anchors, ]
        #   gt_argmax_ious is the serial number of the largest a priori box corresponding to each real box    [num_gt, ]
        # ------------------------------------------------------------------------ #
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)
        
        # ----------------------------------------------------- #
        #   Set to negative samples if less than the threshold value
        #   Set to positive sample if greater than threshold
        #   Each real box corresponds to at least one a priori box
        # ----------------------------------------------------- #
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_ious)>0:
            label[gt_argmax_ious] = 1

        # ----------------------------------------------------- #
        #   Determine whether the number of positive samples is greater than 128, and if so, limit it to 128
        # ----------------------------------------------------- #
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # ----------------------------------------------------- #
        #   Balance positive and negative samples, keeping the total number at 256
        # ----------------------------------------------------- #
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label


class ProposalTargetCreator(object):
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        # ----------------------------------------------------- #
        #   Calculate the degree of overlap between the proposed box and the real box
        # ----------------------------------------------------- #
        iou = bbox_iou(roi, bbox)
        
        if len(bbox)==0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            #---------------------------------------------------------#
            #   Get the most corresponding real box for each suggestion box  [num_roi, ]
            #---------------------------------------------------------#
            gt_assignment = iou.argmax(axis=1)
            #---------------------------------------------------------#
            #   Get the iou of the most corresponding real box for each suggestion box  [num_roi, ]
            #---------------------------------------------------------#
            max_iou = iou.max(axis=1)
            #---------------------------------------------------------#
            #   The label of the real box should be +1 because of the presence of the background
            #---------------------------------------------------------#
            gt_roi_label = label[gt_assignment] + 1

        #----------------------------------------------------------------#
        #   Meet the suggestion box and the real box overlap more than neg_iou_thresh_high as negative samples
        #   Limit the number of positive samples to self.pos_roi_per_image
        #----------------------------------------------------------------#
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        #-----------------------------------------------------------------------------------------------------#
        #   Meet the suggestion box and the real box overlap less than neg_iou_thresh_high more than neg_iou_thresh_low as negative samples
        #   Fix the sum of the number of positive samples and the number of negative samples as self.n_sample
        #-----------------------------------------------------------------------------------------------------#
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
            
        #---------------------------------------------------------#
        #   sample_roi      [n_sample, ]
        #   gt_roi_loc      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        #---------------------------------------------------------#
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox)==0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        return sample_roi, gt_roi_loc, gt_roi_label

class FasterRCNNTrainer(nn.Module):
    def __init__(self, model_train, optimizer):
        super(FasterRCNNTrainer, self).__init__()
        self.model_train    = model_train
        self.optimizer      = optimizer

        self.rpn_sigma      = 1
        self.roi_sigma      = 1

        self.anchor_target_creator      = AnchorTargetCreator()
        self.proposal_target_creator    = ProposalTargetCreator()

        self.loc_normalize_std          = [0.1, 0.1, 0.2, 0.2]

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        pred_loc    = pred_loc[gt_label > 0]
        gt_loc      = gt_loc[gt_label > 0]

        sigma_squared = sigma ** 2
        regression_diff = (gt_loc - pred_loc)
        regression_diff = regression_diff.abs().float()
        regression_loss = torch.where(
                regression_diff < (1. / sigma_squared),
                0.5 * sigma_squared * regression_diff ** 2,
                regression_diff - 0.5 / sigma_squared
            )
        regression_loss = regression_loss.sum()
        num_pos         = (gt_label > 0).sum().float()
        
        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return regression_loss
        
    def forward(self, imgs, bboxes, labels, scale):
        n           = imgs.shape[0]
        img_size    = imgs.shape[2:]
        #-------------------------------#
        #   Get common feature layer
        #-------------------------------#
        base_feature = self.model_train(imgs, mode = 'extractor')

        # -------------------------------------------------- #
        #   Use rpn network to obtain adjustment parameters, scores, suggestion boxes, a priori boxes
        # -------------------------------------------------- #
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.model_train(x = [base_feature, img_size], scale = scale, mode = 'rpn')
        
        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all  = 0, 0, 0, 0
        sample_rois, sample_indexes, gt_roi_locs, gt_roi_labels                 = [], [], [], []
        for i in range(n):
            bbox        = bboxes[i]
            label       = labels[i]
            rpn_loc     = rpn_locs[i]
            rpn_score   = rpn_scores[i]
            roi         = rois[i]
            # -------------------------------------------------- #
            #   Using the real frame and the a priori frame to obtain the prediction results that the proposed frame network should have
            #   Label each a priori box
            #   gt_rpn_loc      [num_anchors, 4]
            #   gt_rpn_label    [num_anchors, ]
            # -------------------------------------------------- #
            gt_rpn_loc, gt_rpn_label    = self.anchor_target_creator(bbox, anchor[0].cpu().numpy())
            gt_rpn_loc                  = torch.Tensor(gt_rpn_loc).type_as(rpn_locs)
            gt_rpn_label                = torch.Tensor(gt_rpn_label).type_as(rpn_locs).long()
            # -------------------------------------------------- #
            #   Calculate the regression loss and classification loss of the proposed box network, respectively
            # -------------------------------------------------- #
            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
  
            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            # ------------------------------------------------------ #
            #   Obtain the prediction results that the classifier network should have using the real frame and the suggested frame
            #   Get three variables, sample_roi, gt_roi_loc, gt_roi_label
            #   sample_roi      [n_sample, ]
            #   gt_roi_loc      [n_sample, 4]
            #   gt_roi_label    [n_sample, ]
            # ------------------------------------------------------ #
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_std)
            sample_rois.append(torch.Tensor(sample_roi).type_as(rpn_locs))
            sample_indexes.append(torch.ones(len(sample_roi)).type_as(rpn_locs) * roi_indices[i][0])
            gt_roi_locs.append(torch.Tensor(gt_roi_loc).type_as(rpn_locs))
            gt_roi_labels.append(torch.Tensor(gt_roi_label).type_as(rpn_locs).long())
            
        sample_rois     = torch.stack(sample_rois, dim=0)
        sample_indexes  = torch.stack(sample_indexes, dim=0)
        roi_cls_locs, roi_scores = self.model_train([base_feature, sample_rois, sample_indexes, img_size], mode = 'head')
        for i in range(n):
            # ------------------------------------------------------ #
            #   According to the type of suggestion box, take out the corresponding regression prediction results
            # ------------------------------------------------------ #
            n_sample = roi_cls_locs.size()[1]
            
            roi_cls_loc     = roi_cls_locs[i]
            roi_score       = roi_scores[i]
            gt_roi_loc      = gt_roi_locs[i]
            gt_roi_label    = gt_roi_labels[i]
            
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc     = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]

            # -------------------------------------------------- #
            #   Calculate the regression loss and classification loss of Classifier network respectively
            # -------------------------------------------------- #
            roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss
            
        losses = [rpn_loc_loss_all/n, rpn_cls_loss_all/n, roi_loc_loss_all/n, roi_cls_loss_all/n]
        losses = losses + [sum(losses)]
        return losses

    def train_step(self, imgs, bboxes, labels, scale, fp16=False, scaler=None):
        self.optimizer.zero_grad()
        if not fp16:
            losses = self.forward(imgs, bboxes, labels, scale)
            losses[-1].backward()
            self.optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                losses = self.forward(imgs, bboxes, labels, scale)

            #----------------------#
            #   Backward Propagation
            #----------------------#
            scaler.scale(losses[-1]).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
        return losses

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
