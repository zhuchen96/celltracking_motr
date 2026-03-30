# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import datetime
import os
import pickle
import subprocess
from argparse import Namespace
from typing import List, Optional
import re
import numpy as np
import math
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

from . import box_ops

if int(re.findall('\d+',(torchvision.__version__[:4]))[-1]) < 7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, _, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False

    elif tensor_list[0].ndim == 4:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size 
        b, _, n, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, n, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2], : img.shape[3]].copy_(img)
            m[:, : img.shape[2], :img.shape[3]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor] = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    def unmasked_tensor(self, index: int):
        tensor = self.tensors[index]

        if not self.mask[index].any():
            return tensor

        h_index = self.mask[index, 0, :].nonzero(as_tuple=True)[0]
        if len(h_index):
            tensor = tensor[:, :, :h_index[0]]

        w_index = self.mask[index, :, 0].nonzero(as_tuple=True)[0]
        if len(w_index):
            tensor = tensor[:, :w_index[0], :]

        return tensor


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)

        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ and 'SLURM_PTY_PORT' not in os.environ:
        # slurm process but not interactive
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    # torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[2]) == 0 and float(torchvision.__version__[3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


class DistributedWeightedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, replacement=True):
        super(DistributedWeightedSampler, self).__init__(dataset, num_replicas, rank, shuffle)

        assert replacement

        self.replacement = replacement

    def __iter__(self):
        iter_indices = super(DistributedWeightedSampler, self).__iter__()
        if hasattr(self.dataset, 'sample_weight'):
            indices = list(iter_indices)

            weights = torch.tensor([self.dataset.sample_weight(idx) for idx in indices])

            g = torch.Generator()
            g.manual_seed(self.epoch)

            weight_indices = torch.multinomial(
                weights, self.num_samples, self.replacement, generator=g)
            indices = torch.tensor(indices)[weight_indices]

            iter_indices = iter(indices.tolist())
        return iter_indices

    def __len__(self):
        return self.num_samples


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, weights, alpha: float = 0.25, gamma: float = 2, query_mask=None, reduction=True, mask=False):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
            
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none",weight=weights)

    if mask:
        return ce_loss.mean(1).sum() / num_boxes

    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if not reduction:
        return loss

    if query_mask is not None:
        loss = torch.stack([l[m].mean(0) for l, m in zip(loss, query_mask)])
        return loss.sum() / num_boxes
    return loss.mean(1).sum() / num_boxes


def nested_dict_to_namespace(dictionary):
    namespace = dictionary
    if isinstance(dictionary, dict):
        namespace = Namespace(**dictionary)
        for key, value in dictionary.items():
            setattr(namespace, key, nested_dict_to_namespace(value))
    return namespace

def nested_dict_to_device(dictionary, device):
    output = {}
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            if not isinstance(value, str):
                output[key] = nested_dict_to_device(value, device)
            else:
                output[key] = value
        return output
    return dictionary.to(device)

def threshold_indices(indices,targets,training_method,target_name,max_ind):
    '''
    indices: output from Hungarian matcher
    max_ind: the number of predictions in current batch
    return_swap_indices: if True, return the indices that indicate the cls/boxes need to be swapped

    This functions thresholds the indices outputted from the matcher.
    The swap_indices tells which boxes/class predictions need to be flipped.
    '''

    for (ind_out,ind_tgt),target in zip(indices,targets):

        for i in range(len(ind_out)):
          
            if ind_out[i] >= max_ind:
                ind_out[i] -= max_ind

                assert target[training_method][target_name]['boxes'][ind_tgt[i]][-1] > 0, 'Currently, this only swaps boxes where divisions have occurred. Object detection should only occur in the first box, not the second.'

                target[training_method][target_name]['boxes'][ind_tgt[i]] = torch.cat((target[training_method][target_name]['boxes'][ind_tgt[i],4:],target[training_method][target_name]['boxes'][ind_tgt[i],:4]),axis=-1)
                target[training_method][target_name]['labels'][ind_tgt[i]] = torch.cat((target[training_method][target_name]['labels'][ind_tgt[i],1:],target[training_method][target_name]['labels'][ind_tgt[i],:1]),axis=-1)

                if 'masks' in target[training_method][target_name]:
                    target[training_method][target_name]['masks'][ind_tgt[i]] = torch.cat((target[training_method][target_name]['masks'][ind_tgt[i],1:],target[training_method][target_name]['masks'][ind_tgt[i],:1]),axis=0)

    return indices, targets


def update_metrics_dict(metrics_dict:dict,acc_dict:dict,loss_dict:dict,weight_dict:dict,i,lr=None):
    '''
    After every iteration, the metrics dict is updated with the current loss and acc for that sample

    metrics_dict: dict
    Stores data for all epochs (metrics + loss)
    acc_dict: dict
    Stores acc info for current iteration
    loss_dict: dict
    Stores loss info for current iteration
    weight_dict: dict
    Stores weights for each loss
    i: int
    Iteration number
    '''
    
    metrics_keys = ['det_bbox_acc','det_mask_acc','track_bbox_acc','track_mask_acc','divisions_bbox_acc','divisions_mask_acc','new_cells_bbox_acc','new_cells_mask_acc','new_cells_not_edge_bbox_acc','new_cells_not_edge_mask_acc']

    if i == 0:
        for metrics_key in metrics_keys:
            if metrics_key in acc_dict.keys(): # add the accuracy info; these are two digits; first is # correct; second is total #
                metrics_dict[metrics_key] = acc_dict[metrics_key]
            else:
                metrics_dict[metrics_key] = np.ones((1,1,2)) * np.nan

        for weight_dict_key in weight_dict.keys(): # add the loss info which is a single number
            metrics_dict[weight_dict_key] = (loss_dict[weight_dict_key].detach().cpu().numpy()[None,None] * weight_dict[weight_dict_key]) if weight_dict_key in loss_dict else np.array(np.nan)[None,None]
        
        if lr is not None:
            metrics_dict['lr'] = lr
        
    else:
        for metrics_key in metrics_keys:
            if metrics_key in acc_dict.keys():
                metrics_dict[metrics_key] = np.concatenate((metrics_dict[metrics_key],acc_dict[metrics_key]),axis=1)
            else:
                metrics_dict[metrics_key] = np.concatenate((metrics_dict[metrics_key],np.ones((1,1,2)) * np.nan),axis=1)

        for weight_dict_key in weight_dict.keys():
            loss_dict_key_loss = (loss_dict[weight_dict_key].detach().cpu().numpy()[None,None] * weight_dict[weight_dict_key]) if weight_dict_key in loss_dict else np.array(np.nan)[None,None]
            metrics_dict[weight_dict_key] = np.concatenate((metrics_dict[weight_dict_key],loss_dict_key_loss),axis=1)

    assert metrics_dict['loss'].shape[0] == 1, 'Only one epoch worth of loss / metric info should be added'

    return metrics_dict

def display_loss(metrics_dict:dict,i,i_total,epoch,dataset):
    '''Print the loss
    
    metrics_dict:dict
    Contains loss / acc info over all epochs
    i: int
    Describes iteration #'''

    display_loss = {}

    for key in metrics_dict.keys():
        if ('loss' in key and not bool(re.search('\d',key)) and key != 'lr') or 'CoMOT' in key:
            display_loss[key] = f'{np.nan if np.isnan(metrics_dict[key][-1]).all() else np.nanmean(metrics_dict[key][-1]):.4f}'

    pad = int(math.log10(i_total))+1
    print(f'{dataset}  Epoch: {epoch} ({i:0{pad}}/{i_total-1})',display_loss)


def save_metrics_pkl(metrics_dict,output_dir,dataset,epoch):

    if not (output_dir / ('metrics_' + dataset + '.pkl')).exists() or epoch == 1:
        with open(output_dir / ('metrics_' + dataset + '.pkl'), 'wb') as f:
            pickle.dump(metrics_dict, f)
    else:
        with open(output_dir / ('metrics_' + dataset + '.pkl'), 'rb') as f:
            loaded_metrics_dict = pickle.load(f)

            for metrics_dict_key in metrics_dict.keys():
                loaded_metrics_dict[metrics_dict_key] = np.concatenate((loaded_metrics_dict[metrics_dict_key],metrics_dict[metrics_dict_key]),axis=0)
        
        with open(output_dir / ('metrics_' + dataset + '.pkl'), 'wb') as f:
            pickle.dump(loaded_metrics_dict, f)



def calc_bbox_acc(acc_dict,outputs,targets,args,calc_mask_acc=True,text=''):
    cls_thresh = args.cls_threshold
    iou_thresh = args.iou_threshold
    TP_bbox = TP_mask = FN = FP = FP_bbox = FP_mask = 0
    for t,target in enumerate(targets):
        indices = target['indices']
        pred_logits = outputs['pred_logits'].sigmoid().detach()[t]

        if target['empty']: # No objects in image so it should be all zero
            FP += int((pred_logits > cls_thresh).sum())
            continue

        FP += sum([1 for ind in range(pred_logits.shape[0]) if (ind not in indices[0] and pred_logits[ind,0] > cls_thresh)])

        pred_boxes = outputs['pred_boxes'].detach()[t]
        tgt_boxes = target['boxes'].detach()

        if 'track_queries_mask' in target:
            assert target['track_queries_mask'].sum() == 0, 'This function calculates detection accuracy only; not tracking'
        assert tgt_boxes[:,4:].sum() == 0, 'All boxes should not contain divisions since only object detection is being done here'

        if 'pred_masks' in outputs and calc_mask_acc:
            pred_masks = outputs['pred_masks'].sigmoid().detach()[t]
            tgt_masks = target['masks'].detach()

        for ind_out, ind_tgt in zip(indices[0],indices[1]):
            if pred_logits[ind_out,0] > cls_thresh:
                iou = box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(pred_boxes[ind_out:ind_out+1,:4]),
                    box_ops.box_cxcywh_to_xyxy(tgt_boxes[ind_tgt:ind_tgt+1,:4]),
                    return_iou_only=True)

                if iou > iou_thresh:
                    TP_bbox += 1
                else:
                    FP_bbox += 1

                if 'pred_masks' in outputs and calc_mask_acc:
                    pred_mask = pred_masks[ind_out:ind_out+1,:1]
                    pred_mask_scaled = F.interpolate(pred_mask, size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False)
                    pred_mask_scaled = (pred_mask_scaled > 0.5) * 1.
                    mask_iou = box_ops.mask_iou(pred_mask_scaled.flatten(1),tgt_masks[ind_tgt:ind_tgt+1,:1].flatten(1))

                    if mask_iou > iou_thresh:
                        TP_mask += 1
                    else:
                        FP_mask += 1
            else:
                FN += 1

    acc_dict[text+'det_bbox_acc'] = np.array((TP_bbox,TP_bbox + FN + FP + FP_bbox),dtype=np.int32)[None,None]

    if 'pred_masks' in outputs:
        acc_dict[text+'det_mask_acc'] = np.array((TP_mask,TP_mask + FN + FP + FP_mask ),dtype=np.int32)[None,None]

    return acc_dict

def calc_track_acc(track_acc_dict,outputs,targets,args, calc_mask_acc=True):
    cls_thresh = args.cls_threshold
    iou_thresh = args.iou_threshold
    TP_bbox = TP_mask = FN = FP = FP_bbox = FP_mask = 0
    div_acc = np.zeros((2),dtype=np.int32)
    div_bbox_acc = np.zeros((2),dtype=np.int32)
    div_mask_acc = np.zeros((2),dtype=np.int32)
    new_cells_acc = np.zeros((2),dtype=np.int32) 
    new_cells_bbox_acc = np.zeros((2),dtype=np.int32) 
    new_cells_mask_acc = np.zeros((2),dtype=np.int32)

    for t,target in enumerate(targets):
        indices = target['indices']
        pred_logits = outputs['pred_logits'][t].sigmoid().detach()
        pred_boxes = outputs['pred_boxes'][t].detach()
        tgt_boxes = target['boxes']

        if target['empty']: # No objects to track
            FP += int((pred_logits[:,0] > cls_thresh).sum())
            continue

        # Coutning False Positives; cells leaving the frame + False Positives added to the frame
        pred_logits_FPs = pred_logits[~target['track_queries_TP_mask'] * target['track_queries_mask'],0]
        FP += int((pred_logits_FPs > cls_thresh).sum())

        if calc_mask_acc and 'pred_masks' in outputs:
            pred_masks = outputs['pred_masks'].sigmoid().detach()[t]
            tgt_masks = target['masks'].detach()

        # Calculate accuracy for new objects detected; FPs or TPs
        for query_id in range(pred_logits.shape[0]):
            if (~target['track_queries_mask'])[query_id]:
                if query_id in indices[0]:
                    ind_loc = torch.where(indices[0] == query_id)[0]
                    if pred_logits[query_id,0] > cls_thresh:
                        iou = box_ops.generalized_box_iou(
                                box_ops.box_cxcywh_to_xyxy(pred_boxes[query_id,:4][None]),
                                box_ops.box_cxcywh_to_xyxy(tgt_boxes[indices[1][ind_loc],:4]),
                                return_iou_only=True
                                )

                        if iou > iou_thresh:
                            TP_bbox += 1
                            new_cells_bbox_acc += 1

                        else:
                            FP_bbox += 1       
                            new_cells_bbox_acc[1] += 1  

                        if calc_mask_acc and 'pred_masks' in outputs:
                            pred_mask = pred_masks[query_id:query_id+1,:1]
                            pred_mask_scaled = F.interpolate(pred_mask, size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False)
                            pred_mask_scaled = (pred_mask_scaled > 0.5) * 1.
                            mask_iou = box_ops.mask_iou(pred_mask_scaled.flatten(1),tgt_masks[indices[1][ind_loc],:1].flatten(1))

                            if mask_iou > iou_thresh:
                                TP_mask += 1
                                new_cells_mask_acc += 1

                            else:
                                FP_mask += 1
                                new_cells_mask_acc[1] += 1

                    else:
                        FN += 1
                        new_cells_acc[1] += 1

                else:
                    if pred_logits[query_id,0] > cls_thresh:
                        FP += 1

        pred_track_logits = pred_logits[target['track_queries_TP_mask']]
        pred_track_boxes = pred_boxes[target['track_queries_TP_mask']]
        box_matching = target['track_query_match_ids']

        if 'pred_masks' in outputs:
            pred_track_masks = pred_masks[target['track_queries_TP_mask']]

        for p,pred_logit in enumerate(pred_track_logits):
            if pred_logit[0] < cls_thresh:
                FN += 1

            else:
                iou = box_ops.generalized_box_iou(
                        box_ops.box_cxcywh_to_xyxy(pred_track_boxes[p:p+1,:4]),
                        box_ops.box_cxcywh_to_xyxy(tgt_boxes[box_matching[p],:4][None]),
                        return_iou_only=True
                    )
                
                if iou > iou_thresh:
                    TP_bbox += 1
                else:
                    FP_bbox += 1

                if calc_mask_acc and 'pred_masks' in outputs:
                    pred_mask = pred_track_masks[p:p+1,:1]
                    pred_mask_scaled = F.interpolate(pred_mask, size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False)
                    pred_mask_scaled = (pred_mask_scaled > 0.5) * 1.
                    mask_iou = box_ops.mask_iou(pred_mask_scaled.flatten(1),tgt_masks[box_matching[p],:1].flatten(1))

                    if mask_iou > iou_thresh:
                        TP_mask += 1
                    else:
                        FP_mask += 1

            # Need to check for divisions
            if pred_logit[1] > cls_thresh and tgt_boxes[box_matching[p],-1] == 0: # Predicted FP division
                FP += 1  
                div_acc[1] += 1           
            elif pred_logit[1] < cls_thresh and tgt_boxes[box_matching[p],-1] > 0: # Predicted FN division
                FN += 1
                div_acc[1] += 1           
            elif pred_logit[1] > cls_thresh and tgt_boxes[box_matching[p],-1] > 0: # Correctly predictly TP division
                iou = box_ops.generalized_box_iou(
                        box_ops.box_cxcywh_to_xyxy(pred_track_boxes[p:p+1,4:]),
                        box_ops.box_cxcywh_to_xyxy(tgt_boxes[box_matching[p],4:][None]),
                        return_iou_only=True
                    )
                # Divided cells were not accounted above so we add one to correct & total column
                if iou > iou_thresh:
                    TP_bbox += 1
                    div_bbox_acc += 1
                else:
                    FP_bbox += 1
                    div_bbox_acc[1] += 1

                if calc_mask_acc and 'pred_masks' in outputs:
                    pred_mask = pred_track_masks[p:p+1,:1]
                    pred_mask_scaled = F.interpolate(pred_mask, size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False)
                    pred_mask_scaled = (pred_mask_scaled > 0.5) * 1.
                    mask_iou = box_ops.mask_iou(pred_mask_scaled.flatten(1),tgt_masks[box_matching[p],:1].flatten(1))

                    if mask_iou > iou_thresh:
                        TP_mask += 1
                        div_mask_acc += 1
                    else:
                        FP_mask += 1
                        div_mask_acc[1] += 1

            elif pred_logit[1] > cls_thresh and tgt_boxes[box_matching[p],-1] == 0: # Predicted TN correctly
                pass

    track_acc_dict['track_bbox_acc'] = np.array((TP_bbox,TP_bbox + FN + FP + FP_bbox),dtype=np.int32)[None,None]
    track_acc_dict['divisions_bbox_acc'] = (div_acc + div_bbox_acc)[None,None]
    track_acc_dict['new_cells_bbox_acc'] = (new_cells_acc + new_cells_bbox_acc)[None,None]

    if calc_mask_acc and 'pred_masks' in outputs:
        track_acc_dict['track_mask_acc'] = np.array((TP_mask,TP_mask + FN + FP + FP_mask),dtype=np.int32)[None,None]
        track_acc_dict['divisions_mask_acc'] = (div_acc + div_mask_acc)[None,None]
        track_acc_dict['new_cells_mask_acc'] = (new_cells_acc + new_cells_mask_acc)[None,None]

    return track_acc_dict

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def point_sample(input, point_coords, **kwargs):
    'Adapted from Detectron2'
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def man_track_ids(targets,training_method:str,input_target_name:str,output_target_name:str = None):

    target_names = ['prev_prev_target','prev_target','cur_target','fut_target']
    target_names = [target_name for target_name in target_names if target_name in targets[0][training_method]]

    output_target_index = target_names.index(output_target_name)
    future_target_names = [target_name for target_name in target_names if target_names.index(target_name) > output_target_index]

    features = ['track_ids','boxes','labels','flexible_divisions','is_touching_edge']
    if 'masks' in targets[0][training_method][output_target_name]:
        features += ['masks']

    for target in targets:
        for target_name in target_names:
            
            if not target[training_method][target_name]['empty']: # when there are no cells, formatting is weird for empty images. need to fix this in the future
                remove_indices = torch.zeros_like(target[training_method][target_name]['track_ids_orig']).bool()
                
                if 'new_cell_ids' in target[training_method][target_name]: # This is necessary for dn_track and dn_track_group; it removes newly detected cells since we only care about tracking here
                    remove_track_ids = target[training_method][target_name]['new_cell_ids']
                    track_ids_orig = target[training_method][target_name]['track_ids_orig']

                    for remove_track_id in remove_track_ids:
                        remove_indices |= (track_ids_orig == remove_track_id)

                for feature in features:
                    target[training_method][target_name][feature] = target[training_method][target_name][feature + '_orig'][~remove_indices].clone()
       
        input_target = target[training_method][input_target_name]
        output_target = target[training_method][output_target_name]

        if input_target['empty'] or output_target['empty']:
            continue
        
        framenb = output_target['framenb']
        prev_track_ids = input_target['track_ids']

        if 'prev_ind' in output_target:
            prev_track_ids = prev_track_ids[output_target['prev_ind'][1]]

        # This is needed if false negatives are added or tracks are removed. So groundtruths exist but need to be ignored
        if 'target_ind_matching' in output_target:
            if output_target['num_FPs'] == 0:
                target_ind_matching = output_target['target_ind_matching']
            else:
                target_ind_matching = output_target['target_ind_matching'][:-output_target['num_FPs']]

        man_track = target[training_method]['man_track'].clone()
        man_track = man_track[(man_track[:,1] <= framenb) * (man_track[:,2] >= framenb)]
        
        # cell_divisions = man_track[:,-1]
        cell_divisions = man_track[:,-1] * (man_track[:,1] == framenb) # only check for divisions that occur in the current frame

        for idx,prev_track_id in enumerate(prev_track_ids):
            if 'target_ind_matching' in output_target and not target_ind_matching[idx]:
                continue # This is necesary for when FN are added; the model is forced to detect the cell instead of tracking it

            if prev_track_id not in output_target['track_ids_orig']: # If cell does not track to next frame
                if prev_track_id in cell_divisions: # check if cell divided

                    div_cur_track_ids = man_track[man_track[:,-1] == prev_track_id,0]

                    if len(div_cur_track_ids) == 2:

                        div_ind_1 = output_target['track_ids'] == div_cur_track_ids[0]
                        div_ind_2 = output_target['track_ids'] == div_cur_track_ids[1]

                        if div_ind_1.sum() == 1 and div_ind_2.sum() == 1:

                            output_target['track_ids'][div_ind_1] = prev_track_id
                            remove_ind = output_target['track_ids'] != div_cur_track_ids[1]         

                            for feature in features:
                                if feature not in ['flexible_divisions','track_ids','is_touching_edge']:
                                    feature_len = output_target[feature].shape[1]
                                    output_target[feature][div_ind_1,feature_len//2:] = output_target[feature][div_ind_2,:feature_len//2]
                                
                                if feature == 'is_touching_edge' and output_target['is_touching_edge'][div_ind_2]:
                                    output_target['is_touching_edge'][div_ind_1] = True

                                output_target[feature] = output_target[feature][remove_ind]

                        elif div_ind_1.sum() == 1 or div_ind_2.sum() == 1:

                            if div_ind_1.sum() == 1:
                                div_ind = div_ind_1
                                div_ind_orig = output_target['track_ids_orig'] == div_cur_track_ids[0]
                            else:
                                div_ind = div_ind_2
                                div_ind_orig = output_target['track_ids_orig'] == div_cur_track_ids[1]
                                div_cur_track_ids = torch.flip(div_cur_track_ids,dims=[0])

                            assert div_ind_orig.sum() == 1
                            output_target['track_ids'][div_ind] = prev_track_id
                            output_target['track_ids_orig'][div_ind_orig] = prev_track_id

                            assert target[training_method]['man_track'][target[training_method]['man_track'][:,0] == prev_track_id,2] == framenb-1

                            # Have mother cell replace daughter cell that is still in frame 
                            dau_cells = target[training_method]['man_track'][target[training_method]['man_track'][:,-1] == div_cur_track_ids[0],0]
                            target[training_method]['man_track'][target[training_method]['man_track'][:,0] == dau_cells,-1] = prev_track_id
                            target[training_method]['man_track'][target[training_method]['man_track'][:,0] == prev_track_id,2] = target[training_method]['man_track'][target[training_method]['man_track'][:,0] == div_cur_track_ids[0],2] 
                            
                            target[training_method]['man_track'][target[training_method]['man_track'][:,0] == div_cur_track_ids[0],1:] = -1 # remove cell from lineage since the mother cell replaced it
                            target[training_method]['man_track'][target[training_method]['man_track'][:,0] == div_cur_track_ids[1],-1] = 0 # remove division track from other cell

                            for future_target_name in future_target_names:
                                fut_target = target[training_method][future_target_name]
                                fut_target['track_ids'][fut_target['track_ids'] == div_cur_track_ids[0]] = prev_track_id
                                fut_target['track_ids_orig'][fut_target['track_ids_orig'] == div_cur_track_ids[0]] = prev_track_id
                            
                    else:
                        raise NotImplementedError

            removed_cells_input = 0 
            removed_cells_output = 0 

            if 'new_cell_ids' in input_target:
                removed_cells_input += len(input_target['new_cell_ids'])
            if 'new_cell_ids' in output_target:
                removed_cells_output += len(output_target['new_cell_ids'])

            assert input_target['boxes_orig'].shape[0] == (input_target['boxes'].shape[0] + (input_target['boxes'][:,-1] > 0).sum().item() + removed_cells_input)
            assert output_target['boxes_orig'].shape[0] == (output_target['boxes'].shape[0] + (output_target['boxes'][:,-1] > 0).sum().item() + removed_cells_output)

    for target_name in target_names:
        if len(target[training_method][target_name]['track_ids'].shape) == 1:
            for track_id in target[training_method][target_name]['track_ids']:
                if track_id not in target[training_method]['man_track'][:,0]:
                    raise NotImplementedError

    return targets


def update_cropped_man_track(target):

    target_names = ['prev_target','cur_target']

    if 'prev_prev_target' in target:
        target_names = ['prev_prev_target'] + target_names

    if 'fut_target' in target:
        target_names += ['fut_target']

    man_track = target['man_track']
    max_cellnb = target['man_track'][-1,0].item()

    for target_name in target_names:

        output_target_index = target_names.index(target_name)
        future_target_names = [output_target_name for output_target_name in target_names if target_names.index(output_target_name) > output_target_index]

        framenb = target[target_name]['framenb']

        track_ids = man_track[(man_track[:,1] <= framenb) * (man_track[:,2] >= framenb),0]

        for track_id in track_ids:

            cell_track = man_track[man_track[:,0] == track_id][0]
            # if cell is not in cropped frame and it wasn't just born
            if track_id not in target[target_name]['track_ids'] and man_track[man_track[:,0] == track_id,1] != -1:

                # If one of cells that divided are not present in frame, assume no division occured
                if cell_track[-1] > 0 and cell_track[1] == framenb:
                    other_dau_id = man_track[(man_track[:,-1] == cell_track[-1]) * (man_track[:,0] != track_id),0].item()

                    man_track[man_track[:,0] == other_dau_id,-1] = 0
                    man_track[man_track[:,0] == track_id,-1] = 0

                    if other_dau_id in target[target_name]['track_ids']:

                        mother_id = cell_track[-1]

                        div_ind = target[target_name]['track_ids'] == other_dau_id
                        div_ind_orig = target[target_name]['track_ids_orig'] == other_dau_id
                        assert div_ind.sum() == 1 and div_ind_orig.sum() == 1
                        target[target_name]['track_ids'][div_ind] = mother_id   
                        target[target_name]['track_ids_orig'][div_ind_orig] = mother_id

                        # Have mother cell replace daughter cell that is still in frame 
                        dau_cells = man_track[man_track[:,-1] == other_dau_id,0]
                        if len(dau_cells) > 0:
                            man_track[man_track[:,0] == dau_cells[0],-1] = mother_id
                            man_track[man_track[:,0] == dau_cells[1],-1] = mother_id
                            
                        man_track[man_track[:,0] == mother_id,2] = man_track[man_track[:,0] == other_dau_id,2] 
                        
                        man_track[man_track[:,0] == other_dau_id,1:] = -1 # remove cell from lineage since the mother cell replaced it
                        man_track[man_track[:,0] == track_id,1] = framenb+1 # remove cell from lineage since the mother cell replaced it

                        for future_target_name in future_target_names:
                            fut_target = target[future_target_name]
                            fut_target['track_ids'][fut_target['track_ids'] == other_dau_id] = mother_id
                            fut_target['track_ids_orig'][fut_target['track_ids_orig'] == other_dau_id] = mother_id
                        
                    else:
                        man_track[man_track[:,0] == other_dau_id,1] = framenb + 1
                        man_track[man_track[:,0] == track_id,1] = framenb + 1

                        if man_track[man_track[:,0] == other_dau_id,2] < man_track[man_track[:,0] == other_dau_id,1]:
                            man_track[man_track[:,0] == other_dau_id,1:] = -1
                            man_track[man_track[:,-1] == other_dau_id,-1] = 0

                else:
                    new_cell = True
                
                    for future_target_name in future_target_names:
                        fut_target = target[future_target_name]
                        fut_framenb = target[future_target_name]['framenb'].item()

                        if track_id in fut_target['track_ids']:

                            if new_cell:
                                max_cellnb += 1

                                exit_framenb = man_track[man_track[:,0] == track_id,2][0]
                                new_cell = torch.tensor([[max_cellnb,fut_framenb,exit_framenb,0]]).to(man_track.device)
                                man_track = torch.cat((man_track,new_cell))
                                man_track[man_track[:,0] == track_id,2] = framenb-1

                                dau_cells = man_track[man_track[:,-1] == track_id,0]

                                if len(dau_cells) > 0:
                                    man_track[man_track[:,0] == dau_cells[0],-1] = max_cellnb
                                    man_track[man_track[:,0] == dau_cells[1],-1] = max_cellnb
                                new_cell = False

                            assert track_id in fut_target['track_ids'] and track_id in fut_target['track_ids_orig']

                            fut_target['track_ids'][fut_target['track_ids'] == track_id] = max_cellnb
                            fut_target['track_ids_orig'][fut_target['track_ids_orig'] == track_id] = max_cellnb
                        else:
                            new_cell = True

                    man_track[man_track[:,0] == track_id,2] = framenb - 1 

                    if track_id in man_track[:,-1]:
                        dau_cells = man_track[man_track[:,-1] == track_id,0]
                        if len(dau_cells) > 0:
                            man_track[man_track[:,0] == dau_cells[0],-1] = 0                                           
                            man_track[man_track[:,0] == dau_cells[1],-1] = 0                                           
                
                if man_track[man_track[:,0] == track_id,2] < man_track[man_track[:,0] == track_id,1]:
                    man_track[man_track[:,0] == track_id,1:] = -1
                    man_track[man_track[:,-1] == track_id,-1] = 0

                    dau_cells = man_track[man_track[:,-1] == track_id,0]
                    if len(dau_cells) > 0:
                        man_track[man_track[:,0] == dau_cells[0],-1] = 0
                        man_track[man_track[:,0] == dau_cells[1],-1] = 0
                    
    target['man_track'] = man_track

    return target


def split_outputs(outputs,target_TM):

    start_ind = target_TM['start_query_ind']
    end_ind = target_TM['end_query_ind']

    new_outputs = {}
    
    if new_outputs is None:
        new_outputs = outputs
    new_outputs['pred_logits'] = outputs['pred_logits'][:,start_ind:end_ind]
    new_outputs['pred_boxes'] = outputs['pred_boxes'][:,start_ind:end_ind]

    if 'pred_masks' in outputs:
        new_outputs['pred_masks'] = outputs['pred_masks'][:,start_ind:end_ind]

    if 'hs_embed' in outputs:
        new_outputs['hs_embed'] = outputs['hs_embed'][:,start_ind:end_ind]

    if 'pred_div_ahead' in outputs:
        new_outputs['pred_div_ahead'] = outputs['pred_div_ahead'][:,start_ind:end_ind]

    if 'aux_outputs' in outputs:

        new_outputs['aux_outputs'] = [{} for _ in range(len(outputs['aux_outputs']))]

        for lid in range(len(outputs['aux_outputs'])):
            new_outputs['aux_outputs'][lid]['pred_logits'] = outputs['aux_outputs'][lid]['pred_logits'][:,start_ind:end_ind]
            new_outputs['aux_outputs'][lid]['pred_boxes'] = outputs['aux_outputs'][lid]['pred_boxes'][:,start_ind:end_ind]

            if 'pred_masks' in outputs['aux_outputs'][lid]:
                new_outputs['aux_outputs'][lid]['pred_masks'] = outputs['aux_outputs'][lid]['pred_masks'][:,start_ind:end_ind]

    return new_outputs


def get_total_time(args):

    total_time = datetime.timedelta()

    if not (args.output_dir / "training_time.txt").exists():
        return str(total_time)

    with open(str(args.output_dir / "training_time.txt"), 'r') as file:
        for line in file:
            if line.strip():  # Check if line is not empty
                if "Total time:" in line:
                    return str(total_time)
                
                _, time_str = line.split(': ')
                hours, minutes, seconds = map(int, time_str.split(':'))
                total_time += datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)

    with open(str(args.output_dir / "training_time.txt"), "a") as f:
        f.write(f"Total time: {total_time}\n")

    return str(total_time)

def create_folders(train_output_folder,val_output_folder,args):

    args.output_dir.mkdir(exist_ok=True)
    (args.output_dir / val_output_folder).mkdir(exist_ok=True)
    (args.output_dir / train_output_folder).mkdir(exist_ok=True)

    (args.output_dir / val_output_folder / 'standard').mkdir(exist_ok=True)
    (args.output_dir / train_output_folder / 'standard').mkdir(exist_ok=True)

    if args.two_stage:
        (args.output_dir / val_output_folder / 'two_stage').mkdir(exist_ok=True)
        (args.output_dir / train_output_folder / 'two_stage').mkdir(exist_ok=True)

        if args.dn_enc:
            (args.output_dir / val_output_folder / 'dn_enc').mkdir(exist_ok=True)
            (args.output_dir / train_output_folder / 'dn_enc').mkdir(exist_ok=True) 

    if args.dn_track and args.tracking:
        (args.output_dir / val_output_folder / 'dn_track').mkdir(exist_ok=True)
        (args.output_dir / train_output_folder / 'dn_track').mkdir(exist_ok=True)     

    if args.dn_track and args.dn_track_group and args.tracking:
        (args.output_dir / val_output_folder / 'dn_track_group').mkdir(exist_ok=True)
        (args.output_dir / train_output_folder / 'dn_track_group').mkdir(exist_ok=True)  

    if args.dn_object:
        (args.output_dir / val_output_folder / 'dn_object').mkdir(exist_ok=True)
        (args.output_dir / train_output_folder / 'dn_object').mkdir(exist_ok=True)   

    if args.CoMOT:
        (args.output_dir / val_output_folder / 'CoMOT').mkdir(exist_ok=True)
        (args.output_dir / train_output_folder / 'CoMOT').mkdir(exist_ok=True)  

    if args.num_OD_layers > 0:
        (args.output_dir / val_output_folder / 'OD').mkdir(exist_ok=True)
        (args.output_dir / train_output_folder / 'OD').mkdir(exist_ok=True)  

def load_model(model_without_ddp,args,param_dicts=None,optimizer=None,lr_scheduler=None):

    if str(args.resume).startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)

    model_state_dict = model_without_ddp.state_dict()
    checkpoint_state_dict = checkpoint['model']
    checkpoint_state_dict = {k.replace('detr.', ''): v for k, v in checkpoint['model'].items()}

    for k, v in checkpoint_state_dict.items():
        if k not in model_state_dict:
            print(f'Where is {k} {tuple(v.shape)}?')

    resume_state_dict = {}
    for k, v in model_state_dict.items():
        if k not in checkpoint_state_dict:
            resume_value = v
            print(f'Load {k} {tuple(v.shape)} from scratch.')
        elif v.shape != checkpoint_state_dict[k].shape:
            checkpoint_value = checkpoint_state_dict[k]
            num_dims = len(checkpoint_value.shape)

            if 'norm' in k:
                resume_value = checkpoint_value.repeat(2)
            elif 'multihead_attn' in k or 'self_attn' in k:
                resume_value = checkpoint_value.repeat(num_dims * (2, ))
            elif 'reference_points' in k and checkpoint_value.shape[0] * 2 == v.shape[0]:
                resume_value = v
                resume_value[:2] = checkpoint_value.clone()
            elif 'linear1' in k or 'query_embed' in k:
                resume_state_dict[k] = v
                print(f'Load {k} {tuple(v.shape)} from scratch.')
                continue
            elif 'linear2' in k or 'input_proj' in k:
                resume_value = checkpoint_value.repeat((2,) + (num_dims - 1) * (1, ))
            elif 'class_embed' in k:
                resume_value = checkpoint_value[list(range(0, 20))]
            else:
                raise NotImplementedError(f"No rule for {k} with shape {v.shape}.")

            print(f"Load {k} {tuple(v.shape)} from resume model "
                    f"{tuple(checkpoint_value.shape)}.")
        elif args.resume_shift_neuron and 'class_embed' in k:
            checkpoint_value = checkpoint_state_dict[k]
            resume_value = checkpoint_value.clone()
            resume_value[:-1] = checkpoint_value[1:].clone()
            resume_value[-2] = checkpoint_value[0].clone()
            print(f"Load {k} {tuple(v.shape)} from resume model and "
                    "shift class embed neurons to start with label=0 at neuron=0.")
        else:
            resume_value = checkpoint_state_dict[k]

        resume_state_dict[k] = resume_value

    if args.masks and args.load_mask_head_from_model is not None:
        checkpoint_mask_head = torch.load(
            args.load_mask_head_from_model, map_location='cpu')

        for k, v in resume_state_dict.items():

            if (('bbox_attention' in k or 'mask_head' in k)
                and v.shape == checkpoint_mask_head['model'][k].shape):
                print(f'Load {k} {tuple(v.shape)} from mask head model.')
                resume_state_dict[k] = checkpoint_mask_head['model'][k]

    model_without_ddp.load_state_dict(resume_state_dict)

    # RESUME OPTIM
    if not args.eval_only and args.resume_optim:
        if 'optimizer' in checkpoint:
            if args.overwrite_lrs:
                for c_p, p in zip(checkpoint['optimizer']['param_groups'], param_dicts):
                    c_p['lr'] = p['lr']

            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            if args.overwrite_lr_scheduler:
                checkpoint['lr_scheduler'].pop('milestones')
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if args.overwrite_lr_scheduler:
                lr_scheduler.step(checkpoint['lr_scheduler']['last_epoch'])

        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"RESUME EPOCH: {args.start_epoch}")

        for file_name in ['metrics_train.pkl','metrics_val.pkl','training_time.txt']:
            if not (args.output_dir / file_name).exists():
                shutil.copyfile(Path(args.resume).parent / file_name, args.output_dir / file_name)

    return model_without_ddp


def add_new_targets_from_main(targets,training_method,target_name):

    keys = ['labels','boxes','track_ids','flexible_divisions','is_touching_edge']

    if 'masks' in targets[0]['main'][target_name]:
        keys += ['masks']

    for target in targets:
        target[training_method] = {'training_method': training_method, target_name: {}}

        for key in keys:
            target[training_method][target_name][key] = target['main'][target_name][key + '_orig'].clone()
            target[training_method][target_name][key + '_orig'] = target['main'][target_name][key + '_orig'].clone()

        target[training_method][target_name]['empty'] = target['main'][target_name]['empty'].clone()

        # This is needed for flex div for OD; could do this for two-stage encoder as well
        target[training_method]['man_track'] = target['main']['man_track'].clone()
        target[training_method][target_name]['framenb'] = target['main'][target_name]['framenb']

    return targets


def plot_loss_and_metrics(datapath):
    
    # Load pickle files with all the data on the losses and metrics
    with open(datapath / 'metrics_train.pkl', 'rb') as f:
        metrics_train = pickle.load(f)

    with open(datapath / 'metrics_val.pkl', 'rb') as f:
        metrics_val = pickle.load(f)

    # Separate the loss and metric data
    losses = [key for key in metrics_train.keys() if 'loss' in key and key != 'loss' and not bool(re.search('\d',key)) and not np.isnan(metrics_train[key]).all()]
    metrics = [key for key in metrics_train.keys() if 'acc' in key and 'not_edge' not in key]

    # Get number of epochs (training and val should have same number of epochs but I have it separate so it won't create an error if that happens)
    epochs = metrics_train['loss'].shape[0]
    epochs_val = metrics_val['loss'].shape[0]

    # Get number of decoder layers 
    num_layers = len(np.unique(np.array([int(re.findall('\d+',key)[0]) for key in metrics_train.keys() if bool(re.findall('\d+',key))])))

    # Get all the training methods used during training
    training_methods = [loss[:-8] for loss in losses if 'loss_ce' in loss]

    # Plot overall loss
    fig,ax = plt.subplots()
    lrs = metrics_train['lr']

    for i in range(lrs.shape[1]):
        ax.plot(np.arange(1,epochs+1),lrs[:,i])
    ax.set_title('Learning Rate Schedule')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('lr')
    fig.savefig(datapath / 'learning_rate.png')

    # Plot Overall Loss
    fig,ax = plt.subplots()
    ax.plot(np.arange(1,epochs+1),np.nanmean(metrics_train['loss'],axis=-1),label='train')
    ax.plot(np.arange(1,epochs_val+1),np.nanmean(metrics_val['loss'],axis=-1),label='val')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(datapath / 'loss_plot_overall.png')

    # Save the overall loss plot with log scale as well
    ax.set_yscale('log')
    plt.savefig(datapath / 'loss_plot_overall_log.png')

    # Plot the individual losses
    fig,ax = plt.subplots(len(training_methods),2,figsize=(10,15))

    min_y = np.inf
    max_y = 0

    for loss in losses:
        if 'CoMOT' in loss or bool(re.search('\d',loss)) or np.isnan(metrics_train[loss]).all():
            continue

        t = [i for i in range(len(training_methods)) if training_methods[i] in loss and (loss[len(training_methods[i]): len(training_methods[i]) + 6] != '_group')][0]
        training_method = training_methods[t]

        plot_epochs_train = np.arange(1,epochs+1)
        plot_epochs_val = np.arange(1,epochs+1)
        plot_epochs_train = plot_epochs_train[~np.isnan(metrics_train[loss]).all(-1)]
        plot_epochs_val = plot_epochs_val[~np.isnan(metrics_val[loss]).all(-1)]
        plot_metrics_train = metrics_train[loss][~np.isnan(metrics_train[loss]).all(-1)]
        plot_metrics_val = metrics_val[loss][~np.isnan(metrics_val[loss]).all(-1)]

        label = loss.replace(training_method + '_','')

        train_loss = np.nanmean(plot_metrics_train,axis=-1)
        val_loss = np.nanmean(plot_metrics_val,axis=-1)
        ax[t,0].plot(plot_epochs_train,train_loss,label=label)
        ax[t,1].plot(plot_epochs_val,val_loss,label=label)
        min_y = min((min_y,min(train_loss),min(val_loss)))
        max_y = max((max_y,max(train_loss),max(val_loss)))


    # Label each plot with correct loss name
    for t,training_method in enumerate(training_methods):
        for i, dataset in enumerate(['train','val']):
            ax[t,i].set_xlabel('Epochs')
            ax[t,i].set_ylabel('Loss')
            ax[t,i].legend()
            ax[t,i].set_title(f'{training_method}: {dataset}')
            ax[t,i].set_ylim(min_y,max_y)

    fig.tight_layout()
    plt.savefig(datapath / 'loss_plot.png')

    # Save individual loss plot with log axis as well
    for t in range(len(training_methods)):
        ax[t,0].set_yscale('log')
        ax[t,1].set_yscale('log')

    plt.savefig(datapath / 'loss_plot_log.png')

    # Remove losses that don't have intermediate losses / CoMOT as a loss if used since it is only an intermediate loss
    losses = ['loss_ce','loss_bbox','loss_giou','loss_mask','loss_dice']

    if not f'main_loss_mask_1' in metrics_train or np.isnan(metrics_train[f'main_loss_mask_1']).all():
        losses = [loss for loss in losses if loss != 'loss_mask']

    if not f'main_loss_dice_1' in metrics_train or np.isnan(metrics_train[f'main_loss_dice_1']).all():
        losses = [loss for loss in losses if loss != 'loss_dice']

    if 'two_stage' in training_methods:
        training_methods.remove('two_stage')

    if 'OD' in training_methods:
        training_methods.remove('OD')

    if 'CoMOT_loss_ce_0' in metrics_train:
        training_methods += ['CoMOT']

    # Plot the auxilliary losses for the intermediate outputs from the decoder
    def plot_aux_losses(losses,metrics_train,metrics_val,training_methods,num_layers):

        for training_method in training_methods:
            
            losses_TM = []

            for key in ['loss_ce','loss_bbox','loss_giou','loss_mask','loss_dice']:
                if key in losses:
                    losses_TM += [training_method + '_' + key]

            fig,ax = plt.subplots(len(losses_TM),2,figsize=(10,len(losses_TM)*3))
            min_y = np.inf
            max_y = 0
            layer_nbs = [''] + [f'_{i:1d}' for i in range(num_layers)]
            for i,loss in enumerate(losses_TM):

                if 'CoMOT_loss_ce' in loss and np.isnan(metrics_train['CoMOT_loss_ce_0']).all():
                    continue
                    
                for layer_nb in layer_nbs:

                    if layer_nb == '' and training_method == 'CoMOT':
                        continue

                    loss_key = loss + layer_nb

                    plot_epochs_train = np.arange(1,epochs+1)
                    plot_epochs_val = np.arange(1,epochs+1)
                    plot_epochs_train = plot_epochs_train[~np.isnan(metrics_train[loss_key]).all(-1)]
                    plot_epochs_val = plot_epochs_val[~np.isnan(metrics_val[loss_key]).all(-1)]
                    plot_metrics_train = metrics_train[loss_key][~np.isnan(metrics_train[loss_key]).all(-1)]
                    plot_metrics_val = metrics_val[loss_key][~np.isnan(metrics_val[loss_key]).all(-1)]
                    train_loss = np.nanmean(plot_metrics_train,axis=-1)
                    val_loss = np.nanmean(plot_metrics_val,axis=-1)
                    ax[i,0].plot(plot_epochs_train,train_loss,label=loss_key)
                    ax[i,1].plot(plot_epochs_val,val_loss,label=loss_key)
                    min_y = min((min_y,np.nanmin(train_loss),np.nanmin(val_loss)))
                    max_y = max((max_y,np.nanmax(train_loss),np.nanmax(val_loss)))

                for d,dataset in enumerate(['train','val']):
                    ax[i,d].set_xlabel('Epochs')
                    ax[i,d].set_ylabel('Loss')
                    ax[i,d].legend()
                    ax[i,d].set_title(f'{dataset}: {loss} {training_method}')
                    ax[i,d].set_ylim(min_y,max_y)

            fig.tight_layout()
            plt.savefig(datapath / (f'aux_loss_{training_method}_plot.png'))

    plot_aux_losses(losses,metrics_train,metrics_val,training_methods,num_layers)

    # Plot acc (Detection accuracy, Tracking Accuracy, Division Accuracy, New Cell Accuracy (cells that enter the frame or FN))
    fig,ax = plt.subplots(1,len(metrics)//2,figsize=(20,5))
    colors = ['k','b']
    metrics_txt = []

    for midx,metric in enumerate(metrics):

        if 'det' in metric:
            i = 0
        elif 'track' in metric:
            i = 1
        elif 'divisions' in metric:
            i = 2
        else:
            i = 3

        replace_words = ['det_','track_','divisions_','new_cells_not_edge_','new_cells_']
        
        train_acc = np.nansum(metrics_train[metric],axis=-2)
        train_acc = train_acc[:,0] / np.maximum(train_acc[:,1],1)
        val_acc = np.nansum(metrics_val[metric],axis=-2)
        val_acc = val_acc[:,0] / np.maximum(val_acc[:,1],1)

        label_metric = metric.replace(replace_words[i],'')
        label_metric = label_metric.replace('_acc','')

        ax[i].plot(np.arange(1,epochs+1),train_acc, color = colors[midx%2],label=label_metric)
        ax[i].plot(np.arange(1,epochs_val+1),val_acc, '--', color = colors[midx%2])

        print(f'{metric}\nTrain: {train_acc[-1]}\nVal: {val_acc[-1]}')
        metrics_txt.append(f'{metric}\nTrain: {train_acc[-1]}\nVal: {val_acc[-1]}\n')

    # Label each plot with the correct metric name
    titles = ['Detection Accuracy', 'Tracking Accuracy', 'Cell Division Accuracy', 'New Cells Accuracy']

    ax[0].set_ylabel('Accuracy')
    for i in range(len(metrics)//2):
        ax[i].set_xlabel('Epochs')
        ax[i].legend()
        ax[i].set_ylim(0,1)

        ax[i].set_title(titles[i])

    plt.savefig(datapath / 'acc_plot.png')
        
    # Save metric values for last epoch in a txt file for an easy way to see the actually metric values (vs looking at a plot)
    with open(str(datapath / 'metrics.txt'),'w') as f:
        for txt in metrics_txt:
            f.write(txt)