"""MOTR ClipMatcher: direct ID-based assignment for track queries, Hungarian for object queries.

Track queries already know which GT cell they are tracking (via track_query_match_ids).
ClipMatcher exploits this: no Hungarian is needed for track queries — just direct assignment.
Hungarian runs only for the object queries (last num_queries slots) on whatever GT cells
remain unmatched after track query assignment.

This is the principled MOTR assignment, replacing the cost-matrix manipulation in
HungarianMatcher that forces the same outcome through constrained Hungarian.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast

from ..util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ..util.misc import point_sample, threshold_indices
from .matcher import batch_dice_loss_jit, batch_sigmoid_ce_loss_jit


class ClipMatcher(nn.Module):
    """MOTR-style matcher.

    Assignment rules:
      - TP track query j  → GT index track_query_match_ids[j] (direct, cost = -1)
      - FP track query j  → no assignment
      - Object queries    → standard Hungarian on remaining unmatched GT cells
      - Division GT       → assigned directly to track query (combined 8D loss)

    The 2×2 Hungarian for daughter slot ordering is handled upstream in
    MOTRTrackingBase.forward() when building new_prev_indices — by the time this
    matcher is called, track_query_match_ids already carries the correct daughter
    GT indices.
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 1.0,
                 cost_giou: float = 1.0, cost_mask: float = 1.0,
                 cost_dice: float = 1.0, num_points: int = 10000,
                 focal_loss: bool = False, focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0, match_masks: bool = False):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.match_masks = match_masks
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs, targets, training_method, target_name):
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # All-empty fast path
        if all(target[training_method][target_name]['empty'] for target in targets):
            ind = [(torch.tensor([]).long(), torch.tensor([]).long())
                   for _ in range(batch_size)]
            for idx_pair, target in zip(ind, targets):
                target[training_method][target_name]['indices'] = idx_pair
            return ind, targets

        has_tqs = (
            'track_query_match_ids' in targets[0][training_method][target_name]
            and training_method != 'dn_object'
        )
        # num_tqs: number of track-query slots (same across batch due to FP padding)
        num_tqs = 0
        if has_tqs:
            num_tqs = int(
                targets[0][training_method][target_name]['track_queries_mask'].sum()
            )

        sizes = [
            0 if t[training_method][target_name]['empty']
            else len(t[training_method][target_name]['labels'])
            for t in targets
        ]

        all_indices = []

        for i, target in enumerate(targets):
            t = target[training_method][target_name]

            if t['empty']:
                all_indices.append((torch.tensor([]).long(), torch.tensor([]).long()))
                continue

            num_gt = sizes[i]
            pred_idx: list[int] = []
            gt_idx: list[int] = []
            matched_gt: set[int] = set()

            # ── Direct assignment for track queries ───────────────────────
            if has_tqs:
                prop_i = 0
                for j in range(num_queries):
                    if t['track_queries_fal_pos_mask'][j]:
                        pass  # FP: skip
                    elif t['track_queries_mask'][j]:
                        tqid = int(t['track_query_match_ids'][prop_i])
                        prop_i += 1
                        pred_idx.append(j)
                        gt_idx.append(tqid)
                        matched_gt.add(tqid)

            # ── Hungarian for object queries on remaining GT ──────────────
            remaining_gt = [g for g in range(num_gt) if g not in matched_gt]

            if remaining_gt:
                obj_logits = outputs['pred_logits'][i, num_tqs:]  # [Q_obj, C]
                obj_boxes  = outputs['pred_boxes'][i, num_tqs:]   # [Q_obj, 8]

                r_labels = t['labels'][remaining_gt]              # [R, 2]
                r_boxes  = t['boxes'][remaining_gt]               # [R, 8]
                keep     = r_labels[:, 1] == 0                   # non-dividing GT

                # Classification cost
                if self.focal_loss:
                    prob = obj_logits.sigmoid()
                    neg_c = ((1 - self.focal_alpha)
                             * (prob ** self.focal_gamma)
                             * (-(1 - prob + 1e-8).log()))
                    pos_c = (self.focal_alpha
                             * ((1 - prob) ** self.focal_gamma)
                             * (-(prob + 1e-8).log()))
                    cost_cls = (pos_c[:, :1][:, r_labels[:, 0]]
                                - neg_c[:, :1][:, r_labels[:, 0]])
                else:
                    prob = obj_logits.softmax(-1)
                    cost_cls = -prob[:, r_labels[:, 0]]

                # L1 box cost (both 4D halves for dividing GT cells)
                cost_box = torch.cdist(obj_boxes[:, :4], r_boxes[:, :4], p=1)
                cost_box[:, keep] += torch.cdist(
                    obj_boxes[:, 4:], r_boxes[:, 4:], p=1)[:, keep]

                # GIoU cost
                cost_iou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(obj_boxes[:, :4]),
                    box_cxcywh_to_xyxy(r_boxes[:, :4]),
                )
                cost_iou[:, keep] += -generalized_box_iou(
                    box_cxcywh_to_xyxy(obj_boxes[:, 4:]),
                    box_cxcywh_to_xyxy(r_boxes[:, 4:]),
                )[:, keep]

                cost_mat = (self.cost_class * cost_cls
                            + self.cost_bbox * cost_box
                            + self.cost_giou * cost_iou)

                # Optional mask cost
                if self.match_masks and 'pred_masks' in outputs:
                    obj_masks = outputs['pred_masks'][i, num_tqs:]      # [Q_obj, 2, H, W]
                    r_masks   = t['masks'][remaining_gt].to(obj_masks)  # [R, 2, H, W]
                    r_masks   = F.interpolate(
                        r_masks, size=obj_masks.shape[-2:],
                        mode="bilinear", align_corners=False)

                    n_pts  = min(self.num_points,
                                 obj_masks.shape[-1] * obj_masks.shape[-2])
                    coords = torch.rand(1, n_pts, 2,
                                       device=obj_masks.device,
                                       dtype=r_masks.dtype)
                    r_s   = point_sample(
                        r_masks, coords.expand(r_masks.shape[0], -1, -1),
                        align_corners=False).squeeze(1)
                    obj_s = point_sample(
                        obj_masks, coords.expand(obj_masks.shape[0], -1, -1),
                        align_corners=False).squeeze(1)

                    with autocast(enabled=False):
                        c_mask = batch_sigmoid_ce_loss_jit(
                            obj_s[:, 0].float(), r_s[:, 0].float())
                        c_dice = batch_dice_loss_jit(
                            obj_s[:, 0].float(), r_s[:, 0].float())
                        c_mask[:, keep] += batch_sigmoid_ce_loss_jit(
                            obj_s[:, 1].float(), r_s[:, 1].float())[:, keep]
                        c_dice[:, keep] += batch_dice_loss_jit(
                            obj_s[:, 1].float(), r_s[:, 1].float())[:, keep]

                    cost_mat = cost_mat + self.cost_mask * c_mask + self.cost_dice * c_dice

                row_ind, col_ind = linear_sum_assignment(cost_mat.cpu().numpy())
                for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                    pred_idx.append(num_tqs + r)
                    gt_idx.append(remaining_gt[c])

            all_indices.append((
                torch.tensor(pred_idx, dtype=torch.int64),
                torch.tensor(gt_idx, dtype=torch.int64),
            ))

        # threshold_indices is a no-op here (ClipMatcher never produces indices >= num_queries)
        all_indices, targets = threshold_indices(
            all_indices, targets, training_method, target_name, max_ind=num_queries)

        ind = []
        for (pi, gi), target in zip(all_indices, targets):
            if target[training_method][target_name]['empty']:
                ind.append((torch.tensor([]).long(), torch.tensor([]).long()))
            else:
                ind.append((pi, gi))

        for idx_pair, target in zip(ind, targets):
            target[training_method][target_name]['indices'] = idx_pair

        return ind, targets


def build_clip_matcher(args):
    return ClipMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        cost_mask=args.set_cost_mask,
        cost_dice=args.set_cost_dice,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        match_masks=args.match_masks,
        num_points=args.num_points,
    )
