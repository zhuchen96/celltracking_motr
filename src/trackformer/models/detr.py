# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..util import box_ops
from ..util.misc import dice_loss, nested_tensor_from_tensor_list, sigmoid_focal_loss, add_new_targets_from_main
from ..util.flex_div import update_early_or_late_track_divisions, update_object_detection

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 focal_loss, focal_alpha, focal_gamma, tracking,args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of
                    available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tracking = tracking
        self.device = args.device
        self.args = args
        self.eval_only = False # used to determine number of boxes, average or total depending on training or evaluating for loss

    def loss_labels_focal(self, outputs, targets, training_method, target_name, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[training_method][target_name]["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full((src_logits.shape[0],src_logits.shape[1],src_logits.shape[2]), self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = ((target_classes == 0) * 1).float()

        weights = torch.ones((src_logits.shape)).to(self.device)

        num_preds = src_logits.shape[1]

        if 'CoMOT' in outputs and outputs['CoMOT_loss_ce']:
            weights = torch.zeros((src_logits.shape)).to(self.device)
            weights[idx] = 1.
            num_preds = num_boxes

        weights[target_classes_onehot[:,:,1] == 1.] *= self.args.div_loss_coef
        weights[target_classes_onehot == 1.] *= self.args.pos_wei_loss_coef

        for t,target in enumerate(targets):
            if not target[training_method][target_name]['empty']:
                weights[t][indices[t][0]][target[training_method][target_name]['is_touching_edge'][indices[t][1]]] *= self.args.touching_edge_loss_coef            
                # We want it to be decisive about flexible divisions so we do not zero the weight here

            if training_method == 'main' and 'track_queries_mask' in target['main']['cur_target'] and not target[training_method][target_name]['empty']:
                num_track_queries = target['main']['cur_target']['track_queries_mask'].sum().cpu()
                det_indices = indices[t][0][(indices[t][0] >= num_track_queries) * (~target[training_method][target_name]['is_touching_edge'][indices[t][1]]).cpu()]
                weights[t][det_indices,0] *= self.args.FN_det_query_loss_coef 

        if training_method == 'two_stage':
            weights[:,:,1] = 0

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes, weights,
            alpha=self.focal_alpha, gamma=self.focal_gamma)

        loss_ce *= num_preds

        losses = {training_method + '_loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, training_method, target_name, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss
           and the GIoU loss targets dicts must contain the key "boxes" containing
           a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
           format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]

        assert (sum(src_boxes < 0) == 0).all(), 'Pred boxes should have positive values only' 

        target_boxes = [t[training_method][target_name]['boxes'][i] for t, (_, i) in zip(targets, indices) if not t[training_method][target_name]['empty']]

        target_boxes = torch.cat(target_boxes,dim=0)

        # For empty chambers, there is a placeholder bbox of zeros that needs to be removed
        keep_not_empty_boxes = target_boxes.sum(-1) > 0
        target_boxes = target_boxes[keep_not_empty_boxes]
        src_boxes = src_boxes[keep_not_empty_boxes]

        keep = target_boxes[:,-1] > 0

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none') 
        loss_bbox[keep,4:] = loss_bbox[keep,4:] * self.args.div_loss_coef
        loss_bbox[~keep,4:] = 0 # If there is no division, we do not care about the second bounding box

        is_touching_edge = torch.cat([t[training_method][target_name]['is_touching_edge'][i] for t, (_,i) in zip(targets, indices) if not t[training_method][target_name]['empty']])
        loss_bbox[is_touching_edge] *= self.args.touching_edge_loss_coef

        flexible_divisions = torch.cat([t[training_method][target_name]['flexible_divisions'][i] for t, (_,i) in zip(targets,indices) if not t[training_method][target_name]['empty']])
        loss_bbox[flexible_divisions] *= self.args.flex_div_loss_coef

        losses = {}
        losses[training_method + '_loss_bbox'] = loss_bbox.sum() / num_boxes
       

        loss_giou_track = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[:,:4]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[:,:4])) 

        if keep.sum() > 0:
            loss_giou_track[:,keep] = loss_giou_track[:,keep] / 2 + box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes[:,4:]),
                box_ops.box_cxcywh_to_xyxy(target_boxes[:,4:]))[:,keep] / 2

        loss_giou = 1 - torch.diag(loss_giou_track)
        loss_giou[keep] = loss_giou[keep] * self.args.div_loss_coef
        loss_giou[is_touching_edge] = loss_giou[is_touching_edge] * self.args.touching_edge_loss_coef
        loss_giou[flexible_divisions] = loss_giou[flexible_divisions] * self.args.flex_div_loss_coef

        losses[training_method + '_loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, training_method, target_name, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of
           dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        target_masks = [t[training_method][target_name]["masks"][i] for t, (_, i) in zip(targets, indices) if not t[training_method][target_name]['empty']]

        target_masks = torch.cat(target_masks,axis=0).to(self.device)
        target_masks,_ = nested_tensor_from_tensor_list(target_masks).decompose()

        src_masks = src_masks[src_idx]

        keep_non_empty_chambers = target_masks.flatten(1).sum(-1) > 0
        target_masks = target_masks[keep_non_empty_chambers]
        src_masks = src_masks[keep_non_empty_chambers]

        # upsample predictions to the target size
        # src_masks = F.interpolate(src_masks, size=target_masks.shape[-2:],mode="bilinear", align_corners=False)
        target_masks = F.interpolate(target_masks.float(), size=src_masks.shape[-2:],mode="area")
        # src_masks = F.interpolate(src_masks, size=target_masks.shape[-2:]) # This is a deterministic way to interpolate

        division_ind = target_masks[:,1].sum(-1).sum(-1) > 0

        weights_mask = torch.ones((src_masks.shape)).to(self.device)
        weights_mask[division_ind] *= self.args.div_loss_coef # increase weight for divisions prev to current        
        weights_mask[~division_ind,1] = 0 # You don't care about the second prediction if the cell did not divide
        weights_mask[target_masks > 0.5] *= self.args.mask_weight_target_cell_coef

        for t,target in enumerate(targets):
            if not target[training_method][target_name]['empty']:
                one_target_mask = target[training_method][target_name]["masks"][indices[t][1]]
                one_target_mask = F.interpolate(one_target_mask.float(), size=src_masks.shape[-2:],mode="area")
                weights_mask[self.sizes[t]:self.sizes[t+1],:,one_target_mask.sum((0,1)) > 0] *= self.args.mask_weight_all_cells_coef

        is_touching_edge = torch.cat([t[training_method][target_name]['is_touching_edge'][i] for t, (_,i) in zip(targets, indices) if not t[training_method][target_name]['empty']])
        weights_mask[is_touching_edge[keep_non_empty_chambers]] *= self.args.touching_edge_loss_coef

        flexible_divisions = torch.cat([t[training_method][target_name]['flexible_divisions'][i] for t, (_,i) in zip(targets,indices) if not t[training_method][target_name]['empty']])
        weights_mask[flexible_divisions[keep_non_empty_chambers]] *= self.args.flex_div_loss_coef

        if division_ind.sum() > 0:
            target_masks = torch.cat((target_masks[:,0],target_masks[division_ind,1]))
            src_masks = torch.cat((src_masks[:,0],src_masks[division_ind,1]))
            weights_mask = torch.cat((weights_mask[:,0],weights_mask[division_ind,1]))

        weights_mask = weights_mask.flatten(1)
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1).float()

        losses = {
            training_method + "_loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes, weights_mask,
            alpha=self.focal_alpha, gamma=self.focal_gamma, mask=True),
            training_method + "_loss_dice": dice_loss(src_masks.sigmoid(), target_masks, num_boxes)
        }

        # print(time.time() - start)

        return losses

    def loss_contrastive(self, outputs, targets, training_method, target_name, indices, num_boxes):
        """NT-Xent contrastive loss on track embeddings. Disabled by default (contrastive_loss_coef=0)."""
        if training_method != 'main' or 'hs_embed' not in outputs:
            return {training_method + '_loss_contrastive': torch.tensor(0., device=self.device)}

        hs = outputs['hs_embed']
        temperature = 0.1
        all_anchors, all_positives = [], []

        for i, target in enumerate(targets):
            tgt = target[training_method][target_name]
            if tgt.get('empty', False) or 'track_query_hs_embeds' not in tgt or 'track_queries_TP_mask' not in tgt:
                continue
            n_track = tgt['track_query_hs_embeds'].shape[0]
            tp_track_mask = tgt['track_queries_TP_mask'][:n_track]
            if tp_track_mask.sum() == 0:
                continue
            tp_idx = tp_track_mask.nonzero(as_tuple=True)[0]
            all_anchors.append(tgt['track_query_hs_embeds'][tp_track_mask].detach())
            all_positives.append(hs[i, tp_idx])

        if len(all_anchors) == 0:
            return {training_method + '_loss_contrastive': torch.tensor(0., device=self.device)}

        anchors   = F.normalize(torch.cat(all_anchors,   dim=0), dim=-1)
        positives = F.normalize(torch.cat(all_positives, dim=0), dim=-1)
        N = anchors.shape[0]
        if N < 2:
            return {training_method + '_loss_contrastive': torch.tensor(0., device=self.device)}

        logits = torch.mm(anchors, positives.t()) / temperature
        labels = torch.arange(N, device=self.device)
        return {training_method + '_loss_contrastive': F.cross_entropy(logits, labels)}

    def loss_div_ahead(self, outputs, targets, training_method, target_name, indices, num_boxes):
        """BCE loss: predict at prev frame whether each tracked cell will divide at cur frame."""
        if training_method != 'main' or 'pred_div_ahead' not in outputs:
            return {training_method + '_loss_div_ahead': torch.tensor(0., device=self.device)}

        pred = outputs['pred_div_ahead']  # [B, N_track+N_obj]
        all_pred, all_gt = [], []

        for i, target in enumerate(targets):
            tgt = target[training_method][target_name]
            if tgt.get('empty', False) or 'track_query_div_ahead_gt' not in tgt or 'track_queries_TP_mask' not in tgt:
                continue
            if 'track_query_hs_embeds' not in tgt:
                continue
            n_track = tgt['track_query_hs_embeds'].shape[0]
            tp_track_mask = tgt['track_queries_TP_mask'][:n_track]
            if tp_track_mask.sum() == 0:
                continue
            tp_idx = tp_track_mask.nonzero(as_tuple=True)[0]
            all_pred.append(pred[i, tp_idx])
            all_gt.append(tgt['track_query_div_ahead_gt'])

        if not all_pred:
            return {training_method + '_loss_div_ahead': torch.tensor(0., device=self.device)}

        pred_cat = torch.cat(all_pred)
        gt_cat   = torch.cat(all_gt)
        return {training_method + '_loss_div_ahead': F.binary_cross_entropy_with_logits(pred_cat, gt_cat)}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, training_method, target_name, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels_focal if self.focal_loss else self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'contrastive': self.loss_contrastive,
            'div_ahead': self.loss_div_ahead,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        return loss_map[loss](outputs, targets, training_method, target_name, indices, num_boxes)



    def forward(self, outputs, targets, losses, training_method='main', CoMOT=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """

        if  self.args.flex_div and training_method in ['main','dn_track','dn_track_group']:
            targets = update_early_or_late_track_divisions(outputs,targets,training_method, 'prev_target','cur_target','fut_target')

        indices, targets = self.matcher(outputs, targets, training_method, 'cur_target')

        if self.args.flex_div and training_method in ['main','dn_object']:
            
            if training_method == 'main':
                num_queries = self.args.num_queries          
            else:
                num_queries = targets[0]['dn_object']['cur_target']['num_queries']

            targets, indices = update_object_detection(outputs,targets,indices,num_queries,training_method,'prev_target','cur_target','fut_target')

        for t,target in enumerate(targets):
            target[training_method]['cur_target']['indices'] = indices[t]

        if self.eval_only:
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum((1 - t[training_method]['cur_target']["labels"]).sum() for t in targets if not t[training_method]['cur_target']["empty"])
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=self.device)
            num_boxes = torch.clamp(num_boxes, min=1)
        else:
            num_boxes = 1 # Set num boxes to 0 so that loss is scaled by number of cells intead of scaling by images

        sizes = []
        self.sizes = [0]
        for target in targets:
            if target[training_method]['cur_target']['empty']:
                sizes.append(0)
                self.sizes.append(0)
            else:
                sizes.append(len(target[training_method]['cur_target']['labels']))
                self.sizes.append(len(target[training_method]['cur_target']['labels']))
    
        for loss in self.losses:
            if sum(sizes) != 0 or (sum(sizes) == 0 and loss == 'labels'): # If two empty chambers, only loss will be computed for labels as there is nothing to computer for the boxes / masks
                losses.update(self.get_loss(loss, outputs, targets, training_method, 'cur_target', indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:

            if self.args.CoMOT and training_method == 'main' and outputs['pred_logits'].shape[1] > self.args.num_queries:
                CoMOT = True
                targets = add_new_targets_from_main(targets,'CoMOT','cur_target')

            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if target[training_method]['cur_target']['track_queries_TP_mask'].sum() > 0:
                    assert target[training_method]['cur_target']['track_queries_TP_mask'].sum() == len(target[training_method]['cur_target']['track_query_match_ids'])

                if CoMOT:
                    aux_outputs['CoMOT_loss_ce'] = self.args.CoMOT_loss_ce
                    aux_outputs['CoMOT'] = True

                if training_method != 'dn_object': # this does not work for flex div and dn_object; if remove flex_div then dn_objects works fine
                    indices,targets = self.matcher(aux_outputs, targets, training_method, 'cur_target')

                for loss in self.losses:
                    if (loss == 'masks' and 'pred_masks' not in aux_outputs) or (sum(sizes) == 0 and loss != 'labels'):
                        continue

                    l_dict = self.get_loss(loss, aux_outputs, targets, training_method, 'cur_target', indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if CoMOT:

                    aux_outputs_without_track = {}
                    for key in ['pred_logits', 'pred_boxes']:
                        aux_outputs_without_track[key] = aux_outputs[key][:,-self.args.num_queries:].clone()

                    if 'pred_masks' in aux_outputs:
                        aux_outputs_without_track['pred_masks'] = aux_outputs['pred_masks'][:,-self.args.num_queries:].clone()

                    indices,targets = self.matcher(aux_outputs_without_track, targets, 'CoMOT', 'cur_target')
                    aux_outputs['CoMOT_indices'] = indices

                    for loss in self.losses:
                        if (loss == 'masks' and 'pred_masks' not in aux_outputs) or (sum(sizes) == 0 and loss != 'labels') or (loss == 'labels' and not self.args.CoMOT_loss_ce):
                            continue

                        l_dict = self.get_loss(loss, aux_outputs_without_track, targets, 'CoMOT', 'cur_target', indices, num_boxes)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        if 'two_stage' in outputs:# and sum(sizes) > 0:
            enc_outputs = outputs['two_stage']
            targets = add_new_targets_from_main(targets,'two_stage','cur_target')

            indices, targets = self.matcher(enc_outputs, targets,'two_stage','cur_target')
            outputs['two_stage']['indices'] = indices
            
            for loss in self.losses:
                if (sum(sizes) == 0 and loss != 'labels') or (loss == 'masks' and 'pred_masks' not in enc_outputs):
                    continue

                l_dict = self.get_loss(loss, enc_outputs, targets, 'two_stage', 'cur_target', indices, num_boxes)
                l_dict = {k: v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'OD' in outputs:
            OD_outputs = outputs['OD']
            targets = add_new_targets_from_main(targets,'OD','cur_target')
            
            indices, targets = self.matcher(OD_outputs, targets,'OD','cur_target')

            if False:#self.args.flex_div: # was not able to implement flexible division for object detection in the  first layer
                num_queries = self.args.num_queries          
                targets, indices = update_object_detection(OD_outputs,targets,indices,num_queries,'OD','prev_target','cur_target','fut_target')

            outputs['OD']['indices'] = indices
            
            for t,target in enumerate(targets):
                target['OD']['cur_target']['indices'] = indices[t]
            
            for loss in self.losses:
                if (sum(sizes) == 0 and loss != 'labels') or (loss == 'masks' and 'pred_masks' not in OD_outputs):
                    continue

                l_dict = self.get_loss(loss, OD_outputs, targets, 'OD', 'cur_target', indices, num_boxes)
                l_dict = {k: v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses