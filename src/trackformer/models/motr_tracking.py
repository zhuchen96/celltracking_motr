"""
MOTR-based cell tracking for Cell-TRACTR.

Replaces TrackFormer's DETRTrackingBase with MOTR's architecture:
  1. Per-track Instances state (not rebuilt each frame from scratch)
  2. ClipMatcher: ID-based inheritance + Hungarian only for new/untracked cells
  3. QIMv2: self-attention across all track queries between frames
  4. MemoryBank: temporal attention over rolling frame history
  5. Two-pass design: detection pass caches encoder; tracking pass reuses it

Cell-specific features preserved:
  - Division handling (parent → two daughters)
  - Mask head (segmentation)
  - DN-track and DN-track-group denoising
  - CTC evaluation
  - Division-ahead prediction
"""

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .structures import Instances
from ..util import box_ops
from ..util import misc as utils
from ..util.flex_div import update_early_or_late_track_divisions


# ---------------------------------------------------------------------------
# Positional embedding helper
# ---------------------------------------------------------------------------

def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    """Convert N-d normalised positions to sinusoidal embeddings."""
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb


# ---------------------------------------------------------------------------
# QIMv2 – Query Interaction Module v2
# ---------------------------------------------------------------------------

class QIMv2(nn.Module):
    """Self-attention + FFN update of track embeddings between frames.

    Allows all active track queries to attend to one another so the model can
    reason about cell–cell relationships (e.g. two cells about to divide).
    Updates both the content embedding and, optionally, the positional embedding.
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0,
                 update_query_pos: bool = True, score_thr: float = 0.5):
        super().__init__()
        self.update_query_pos = update_query_pos
        self.score_thr = score_thr

        self.self_attn = nn.MultiheadAttention(d_model, 8, dropout=dropout)

        # Content FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(True)

        # Optional positional FFN
        if update_query_pos:
            self.linear_pos1 = nn.Linear(d_model, d_ffn)
            self.linear_pos2 = nn.Linear(d_ffn, d_model)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        # Feature (query_pos) FFN
        self.linear_feat1 = nn.Linear(d_model, d_ffn)
        self.linear_feat2 = nn.Linear(d_ffn, d_model)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances

        d = track_instances.output_embedding.shape[-1]
        pos_feats = d // 4  # num_pos_feats for sinusoidal embedding

        is_pos = track_instances.scores > self.score_thr

        # Update ref_pts for positive tracks to their latest predicted box
        track_instances.ref_pts[is_pos] = track_instances.pred_boxes.detach().clone()[is_pos]

        out_embed = track_instances.output_embedding
        query_feat = track_instances.query_pos
        query_pos = pos2posemb(track_instances.ref_pts, num_pos_feats=pos_feats)
        q = k = query_pos + out_embed

        # Self-attention
        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[is_pos] = query_feat[is_pos]

        return track_instances


# ---------------------------------------------------------------------------
# MemoryBank – temporal attention over rolling per-track history
# ---------------------------------------------------------------------------

class MemoryBank(nn.Module):
    """Rolling K-frame history per track with temporal attention.

    Each track slot maintains a memory buffer of its last K output embeddings.
    Before entering the next decoder step, the current embedding attends to the
    K historical embeddings, enriching it with temporal context.
    """

    def __init__(self, d_model: int, d_ffn: int, mem_bank_len: int = 4,
                 save_thresh: float = 0.5, with_spatial_attn: bool = False):
        super().__init__()
        self.max_his_length = mem_bank_len
        self.save_thresh = save_thresh
        self.save_period = 3

        self.save_proj = nn.Linear(d_model, d_model)

        self.temporal_attn = nn.MultiheadAttention(d_model, 8, dropout=0)
        self.temporal_fc1 = nn.Linear(d_model, d_ffn)
        self.temporal_fc2 = nn.Linear(d_ffn, d_model)
        self.temporal_norm1 = nn.LayerNorm(d_model)
        self.temporal_norm2 = nn.LayerNorm(d_model)

        self.track_cls = nn.Linear(d_model, 1)

        self.spatial_attn = None
        if with_spatial_attn:
            self.spatial_attn = nn.MultiheadAttention(d_model, 8, dropout=0)
            self.spatial_fc1 = nn.Linear(d_model, d_ffn)
            self.spatial_fc2 = nn.Linear(d_ffn, d_model)
            self.spatial_norm1 = nn.LayerNorm(d_model)
            self.spatial_norm2 = nn.LayerNorm(d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def update(self, track_instances: Instances) -> None:
        embed = track_instances.output_embedding[:, None]  # (N, 1, d)
        scores = track_instances.scores
        mem_padding_mask = track_instances.mem_padding_mask
        device = embed.device

        save_period = track_instances.save_period
        if self.training:
            saved_idxes = scores > 0
        else:
            saved_idxes = (save_period == 0) & (scores > self.save_thresh)
            save_period[save_period > 0] -= 1
            save_period[saved_idxes] = self.save_period

        saved_embed = embed[saved_idxes]
        if len(saved_embed) > 0:
            prev_embed = track_instances.mem_bank[saved_idxes]
            save_embed = self.save_proj(saved_embed)
            mem_padding_mask[saved_idxes] = torch.cat([
                mem_padding_mask[saved_idxes, 1:],
                torch.zeros((len(saved_embed), 1), dtype=torch.bool, device=device)
            ], dim=1)
            track_instances.mem_bank = track_instances.mem_bank.clone()
            track_instances.mem_bank[saved_idxes] = torch.cat([prev_embed[:, 1:], save_embed], dim=1)

    def _forward_temporal_attn(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances

        key_padding_mask = track_instances.mem_padding_mask
        valid_idxes = key_padding_mask[:, -1] == 0  # tracks that have been saved at least once
        embed = track_instances.output_embedding[valid_idxes]

        if len(embed) > 0:
            prev_embed = track_instances.mem_bank[valid_idxes]
            kpm = key_padding_mask[valid_idxes]
            embed2 = self.temporal_attn(
                embed[None],
                prev_embed.transpose(0, 1),
                prev_embed.transpose(0, 1),
                key_padding_mask=kpm,
            )[0][0]
            embed = self.temporal_norm1(embed + embed2)
            embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))
            embed = self.temporal_norm2(embed + embed2)
            track_instances.output_embedding = track_instances.output_embedding.clone()
            track_instances.output_embedding[valid_idxes] = embed

        return track_instances

    def _forward_spatial_attn(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0 or self.spatial_attn is None:
            return track_instances

        embed = track_instances.output_embedding
        d = embed.shape[-1]
        query_pos = track_instances.query_pos[:, :d]
        k = q = embed + query_pos
        embed2 = self.spatial_attn(q[:, None], k[:, None], embed[:, None])[0][:, 0]
        embed = self.spatial_norm1(embed + embed2)
        embed2 = self.spatial_fc2(F.relu(self.spatial_fc1(embed)))
        embed = self.spatial_norm2(embed + embed2)
        track_instances.output_embedding = embed
        return track_instances

    def forward(self, track_instances: Instances, update_bank: bool = True) -> Instances:
        track_instances = self._forward_temporal_attn(track_instances)
        if update_bank:
            self.update(track_instances)
        if self.spatial_attn is not None:
            track_instances = self._forward_spatial_attn(track_instances)
        track_instances.track_scores = self.track_cls(track_instances.output_embedding)[..., 0]
        return track_instances


# ---------------------------------------------------------------------------
# RuntimeTrackerBase – inference-time track lifecycle
# ---------------------------------------------------------------------------

class RuntimeTrackerBase:
    """Score-threshold track lifecycle at inference.

    Assigns global IDs to new detections and kills tracks that have been
    missing for more than `miss_tolerance` consecutive frames.
    """

    def __init__(self, score_thresh: float = 0.5, filter_score_thresh: float = 0.4,
                 miss_tolerance: int = 5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances) -> None:
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0

        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


# ---------------------------------------------------------------------------
# MOTRTrackingBase – main MOTR tracking mixin
# ---------------------------------------------------------------------------

class MOTRTrackingBase(nn.Module):
    """MOTR-style cell tracking mixin.

    Drop-in replacement for DETRTrackingBase that implements:
      - Instances-based per-track state management
      - ClipMatcher (ID-based + Hungarian) for training-time matching
      - QIMv2 for inter-frame track embedding refinement
      - MemoryBank for temporal attention
      - Two-pass forward (detection encoder cached for tracking decoder)

    Cell-specific features preserved:
      - Division handling (FN division boxes, separate_divided_cells)
      - DN-track / DN-track-group denoising tracks
      - Division-ahead prediction
      - Mask initialisation from predicted masks
    """

    def __init__(self,
                 matcher,
                 backprop_prev_frame: bool = False,
                 dn_track: bool = False,
                 dn_track_FPs: bool = False,
                 dn_track_l1: float = 0.0,
                 dn_track_l2: float = 0.0,
                 dn_enc: bool = False,
                 refine_div_track_queries: bool = False,
                 no_data_aug: bool = False,
                 flex_div: bool = False,
                 use_prev_prev_frame: bool = True,
                 num_queries: int = 200,
                 dn_track_group: bool = False,
                 tgt_noise: float = 0.1,
                 # MOTR-specific
                 mem_bank_len: int = 4,
                 score_thresh: float = 0.5,
                 qim_update_query_pos: bool = True,
                 qim_dropout: float = 0.1,
                 det_score_thresh: float = 0.5,
                 ):
        # ---- keep all original DETRTrackingBase attributes ----
        self._matcher = matcher
        self._backprop_prev_frame = backprop_prev_frame
        self.num_queries = num_queries
        self.dn_track = dn_track
        self.dn_track_FPs = dn_track_FPs
        self.dn_track_l1 = dn_track_l1
        self.dn_track_l2 = dn_track_l2
        self.dn_enc = dn_enc
        self.refine_div_track_queries = refine_div_track_queries
        self.no_data_aug = no_data_aug
        self.copy_dict_keys = ['labels', 'boxes', 'masks', 'track_ids', 'flexible_divisions',
                               'is_touching_edge', 'empty', 'framenb', 'labels_orig', 'boxes_orig',
                               'masks_orig', 'track_ids_orig', 'flexible_divisions_orig',
                               'is_touching_edge_orig']
        self.eval_prev_prev_frame = False
        self.dn_track_group = dn_track_group
        self.tgt_noise = tgt_noise
        self.use_prev_prev_frame = use_prev_prev_frame
        self.flex_div = flex_div
        self.train_model = False

        if self.dn_track:
            self.dn_track_embedding = nn.Embedding(1, self.hidden_dim)

        # ---- MOTR components ----
        self.qim = QIMv2(
            d_model=self.hidden_dim,
            d_ffn=self.hidden_dim * 4,
            dropout=qim_dropout,
            update_query_pos=qim_update_query_pos,
            score_thr=score_thresh,
        )

        self.memory_bank = MemoryBank(
            d_model=self.hidden_dim,
            d_ffn=self.hidden_dim * 4,
            mem_bank_len=mem_bank_len,
            save_thresh=score_thresh,
        )
        self.mem_bank_len = mem_bank_len
        self.det_score_thresh = det_score_thresh

        # Inference tracker
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=max(score_thresh - 0.1, 0.1),
        )

        # Learnable detection-pass queries (separate pool, same size as num_queries)
        # Used only in the detection pass; training pass reuses backbone/encoder cache.
        self._det_track_instances = None  # initialised in _generate_empty_tracks

    def train(self, mode: bool = True):
        return super().train(mode)

    # ------------------------------------------------------------------
    # Instances management
    # ------------------------------------------------------------------

    def _generate_empty_tracks(self) -> Instances:
        """Create a fresh Instances for the start of a clip."""
        track_instances = Instances((1, 1))
        n = self.num_queries
        d = self.hidden_dim
        device = next(self.parameters()).device

        # Learnable initial reference points and content embeddings (DAB-DETR)
        track_instances.ref_pts = self.refpoint_embed.weight.detach().clone()
        track_instances.query_pos = self.tgt_embed.weight.detach().clone()

        track_instances.output_embedding = torch.zeros((n, d), device=device)
        track_instances.obj_idxes = torch.full((n,), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((n,), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((n,), dtype=torch.long, device=device)
        track_instances.iou = torch.ones((n,), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((n,), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((n,), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((n, 4), device=device)
        track_instances.pred_logits = torch.zeros((n, self.num_classes + 2), device=device)

        # MemoryBank fields
        track_instances.mem_bank = torch.zeros((n, self.mem_bank_len, d), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((n, self.mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((n,), dtype=torch.float32, device=device)

        return track_instances

    # ------------------------------------------------------------------
    # MOTR-style ID-based matching + Hungarian for new cells
    # ------------------------------------------------------------------

    def _motr_match(self, track_instances: Instances, gt_target: dict, prev_out: dict,
                    prev_indices, i: int) -> tuple:
        """Match track Instances to GT for one frame using MOTR-style matching.

        Step 1: Existing tracks matched by GT track ID (no Hungarian needed).
        Step 2: Hungarian matching for unmatched detection slots vs. new GT cells.

        Returns:
            matched_src_indices: LongTensor  [M]
            matched_tgt_indices: LongTensor  [M]
        """
        device = track_instances.obj_idxes.device

        if gt_target.get('empty', False) or 'track_ids' not in gt_target or len(gt_target['track_ids']) == 0:
            return torch.zeros(0, dtype=torch.long, device=device), \
                   torch.zeros(0, dtype=torch.long, device=device)

        gt_track_ids = gt_target['track_ids']

        # --- Step 1: ID inheritance ---
        track_instances.matched_gt_idxes[:] = -1
        ids_i, ids_j = torch.where(track_instances.obj_idxes[:, None] == gt_track_ids)
        track_instances.matched_gt_idxes[ids_i] = ids_j

        full_idxes = torch.arange(len(track_instances), device=device)
        unmatched_track_idxes = full_idxes[track_instances.obj_idxes == -1]

        already_matched_tgt = track_instances.matched_gt_idxes[track_instances.matched_gt_idxes >= 0]
        tgt_state = torch.zeros(len(gt_track_ids), device=device)
        tgt_state[already_matched_tgt] = 1
        untracked_tgt_idxes = torch.where(tgt_state == 0)[0]

        new_src = torch.zeros(0, dtype=torch.long, device=device)
        new_tgt = torch.zeros(0, dtype=torch.long, device=device)

        # --- Step 2: Hungarian for new cells (unmatched slots vs. untracked GT) ---
        if len(unmatched_track_idxes) > 0 and len(untracked_tgt_idxes) > 0:
            # Build unmatched outputs sub-tensor
            unmatched_pred_logits = track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0)
            unmatched_pred_boxes = track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0)

            unmatched_outputs = {
                'pred_logits': unmatched_pred_logits,
                'pred_boxes': unmatched_pred_boxes,
            }

            # Build a targets list compatible with HungarianMatcher
            sub_target = {k: gt_target[k] for k in gt_target if k not in ('indices', 'prev_ind')}
            sub_target['boxes'] = gt_target['boxes'][untracked_tgt_idxes]
            sub_target['labels'] = gt_target['labels'][untracked_tgt_idxes]
            if 'masks' in gt_target:
                sub_target['masks'] = gt_target['masks'][untracked_tgt_idxes]
            if 'track_ids' in gt_target:
                sub_target['track_ids'] = gt_track_ids[untracked_tgt_idxes]
            if 'flexible_divisions' in gt_target:
                sub_target['flexible_divisions'] = gt_target['flexible_divisions'][untracked_tgt_idxes]
            if 'is_touching_edge' in gt_target:
                sub_target['is_touching_edge'] = gt_target['is_touching_edge'][untracked_tgt_idxes]
            # Mark as detection (no track queries)
            sub_target['track_queries_mask'] = torch.zeros(len(unmatched_track_idxes),
                                                            dtype=torch.bool, device=device)
            sub_target['empty'] = False

            # Wrap in a single-batch targets list
            fake_target = {'main': {'cur_target': sub_target}}
            new_indices, _ = self._matcher(unmatched_outputs, [fake_target], 'main', 'cur_target')
            if len(new_indices[0][0]) > 0:
                new_src_local, new_tgt_local = new_indices[0]
                new_src = unmatched_track_idxes[new_src_local]
                new_tgt = untracked_tgt_idxes[new_tgt_local]

                # Assign IDs to newly matched tracks
                track_instances.obj_idxes[new_src] = gt_track_ids[new_tgt].long()
                track_instances.matched_gt_idxes[new_src] = new_tgt

        # Collect all matched (src, tgt) pairs
        prev_src = full_idxes[track_instances.matched_gt_idxes >= 0]
        prev_tgt = track_instances.matched_gt_idxes[prev_src]

        return prev_src, prev_tgt

    # ------------------------------------------------------------------
    # Post-process: update track Instances after one decoder forward
    # ------------------------------------------------------------------

    def _post_process_single_image(self, frame_out: dict, track_instances: Instances,
                                   gt_target: dict, prev_out: dict, prev_indices,
                                   frame_idx: int, is_last: bool) -> Instances:
        """Update track Instances from decoder outputs, then QIM + MemoryBank."""
        device = track_instances.obj_idxes.device
        n_track = len(track_instances)

        # Scores from last decoder layer
        with torch.no_grad():
            if self.training:
                scores = frame_out['pred_logits'][0, :n_track].sigmoid().max(dim=-1).values
            else:
                scores = frame_out['pred_logits'][0, :n_track, 0].sigmoid()

        track_instances.scores = scores
        track_instances.pred_logits = frame_out['pred_logits'][0, :n_track]
        track_instances.pred_boxes = frame_out['pred_boxes'][0, :n_track]
        track_instances.output_embedding = frame_out['hs_embed'][0, :n_track]

        if self.training:
            # MOTR-style matching
            self._motr_match(track_instances, gt_target, prev_out, prev_indices, 0)
        else:
            self.track_base.update(track_instances)

        # MemoryBank: temporal attention + bank update
        track_instances = self.memory_bank(track_instances)

        # QIMv2: refine track embeddings via self-attention between all active tracks
        if not is_last:
            track_instances = self.qim(track_instances)

        return track_instances

    # ------------------------------------------------------------------
    # Build target dicts from Instances (for existing criterion)
    # ------------------------------------------------------------------

    def _instances_to_targets(self, track_instances: Instances, gt_target: dict,
                              num_FPs: int = 0):
        """Populate the fields that DETRTrackingBase/SetCriterion expect.

        Derives track_queries_TP_mask, target_ind_matching, track_query_match_ids,
        track_query_hs_embeds, track_query_boxes from the MOTR Instances state.
        """
        device = track_instances.obj_idxes.device

        tp_mask = track_instances.matched_gt_idxes >= 0  # True = this slot is a TP
        leaving_mask = (track_instances.obj_idxes >= 0) & ~tp_mask

        gt_target['target_ind_matching'] = torch.cat([
            tp_mask,
            torch.zeros(num_FPs, dtype=torch.bool, device=device)
        ])
        gt_target['cells_leaving_mask'] = torch.cat([
            leaving_mask,
            torch.zeros(num_FPs, dtype=torch.bool, device=device)
        ])
        gt_target['track_query_match_ids'] = track_instances.matched_gt_idxes[tp_mask]

        # Division-ahead GT: whether the matched cell will divide at this frame
        if len(gt_target['track_query_match_ids']) > 0 and 'labels' in gt_target:
            gt_target['track_query_div_ahead_gt'] = (
                gt_target['labels'][gt_target['track_query_match_ids'], 1] == 1
            ).float()
        else:
            gt_target['track_query_div_ahead_gt'] = torch.tensor([], device=device)

        gt_target['track_queries_TP_mask'] = torch.cat([
            gt_target['target_ind_matching'],
            torch.zeros(self.num_queries, dtype=torch.bool, device=device)
        ])

        gt_target['track_queries_mask'] = torch.cat([
            torch.ones(len(track_instances) + num_FPs, dtype=torch.bool, device=device),
            torch.zeros(self.num_queries, dtype=torch.bool, device=device)
        ])

        gt_target['track_queries_fal_pos_mask'] = torch.cat([
            ~gt_target['target_ind_matching'],
            torch.zeros(self.num_queries, dtype=torch.bool, device=device)
        ])

        gt_target['num_FPs'] = num_FPs
        gt_target['num_queries'] = len(track_instances) + num_FPs + self.num_queries

        # Pass track embeddings and boxes for next-frame initialisation
        gt_target['track_query_hs_embeds'] = track_instances.output_embedding
        gt_target['track_query_boxes'] = track_instances.ref_pts

        return gt_target

    # ------------------------------------------------------------------
    # Utility helpers (preserved from DETRTrackingBase)
    # ------------------------------------------------------------------

    def calc_num_FPs(self, targets, target_name):
        num_cells = torch.tensor([len(target['main'][target_name]['prev_ind'][0]) for target in targets])
        num_cells_and_FPs = num_cells.clone()
        max_FPs = torch.ceil(torch.sqrt(num_cells)).long()

        for t in range(len(targets)):
            if torch.rand(1) > 0.25:
                num_cells_and_FPs[t] += torch.randint(0, max(max_FPs[t], 1), (1,)).item()

        max_track = max(num_cells_and_FPs)
        num_FPs = max_track - num_cells

        for t, target in enumerate(targets):
            target['main'][target_name]['num_FPs'] = num_FPs[t]
            target['main'][target_name]['max_FPs'] = max_FPs[t]

    def get_FP_boxes(self, target, target_name, i, prev_out):
        prev_out_ind_uni = torch.unique(target[target_name]['prev_ind'][0])
        prev_out_ind_uni_all = torch.unique(target[target_name]['prev_ind_orig'][0])

        not_prev_out_ind = torch.randperm(prev_out['pred_boxes'].shape[1])
        not_prev_out_ind = [ind.item() for ind in not_prev_out_ind if ind not in prev_out_ind_uni_all]

        random_false_out_ind = []
        FP_boxes = []
        FPs_hs = []

        for j in range(target[target_name]['num_FPs']):
            if j < len(prev_out_ind_uni) and j < target[target_name]['max_FPs'] and target_name == 'cur_target':
                box = prev_out['pred_boxes'][i, prev_out_ind_uni[j]].detach().clone()[:4]
                if target['training_method'] == 'main':
                    l_1, l_2 = 0.2, 0.1
                else:
                    l_1, l_2 = self.dn_track_l1, self.dn_track_l2
                box = box_ops.add_noise_to_boxes(box, l_1, l_2)
            else:
                box = torch.zeros((4)).to(self.device)

            if j < len(not_prev_out_ind):
                random_false_out_ind.append(not_prev_out_ind[j])
            else:
                random_ind = torch.randint(0, len(not_prev_out_ind), (1,))[0]
                random_false_out_ind.append(int(not_prev_out_ind[random_ind]))

            if j < len(prev_out_ind_uni) and target['training_method'] == 'main' and box.sum() > 0 and torch.rand(1) > 0.5:
                FP_hs = prev_out['hs_embed'][i, prev_out_ind_uni[j]].detach().clone()
                FP_hs = torch.normal(0, 0.25, size=FP_hs.shape, device=self.device) + FP_hs
                FPs_hs.append(FP_hs)
            else:
                FPs_hs.append(prev_out['hs_embed'][i, random_false_out_ind[-1]].detach().clone())

            FP_boxes.append(box)

        FP_boxes = torch.stack(FP_boxes)
        FPs_hs = torch.stack(FPs_hs)

        target[target_name]['prev_ind'][0] = torch.tensor(
            target[target_name]['prev_ind'][0].tolist() + random_false_out_ind
        ).long()

        return FP_boxes, FPs_hs

    def update_target(self, target, target_name, index):
        cur_target = target[target_name]
        man_track = target['man_track']

        track_id = cur_target['track_ids'][index]
        track_id_1, track_id_2 = man_track[man_track[:, -1] == track_id, 0]

        track_id_1_orig = cur_target['track_ids_orig'][cur_target['boxes'][index, :4].eq(cur_target['boxes_orig'][:, :4]).all(-1)][0]
        track_id_2_orig = cur_target['track_ids_orig'][cur_target['boxes'][index, 4:].eq(cur_target['boxes_orig'][:, :4]).all(-1)][0]

        if track_id_1 != track_id_1_orig:
            track_id_1, track_id_2 = track_id_2, track_id_1

        cur_target['track_ids'] = torch.cat((cur_target['track_ids'][:index],
                                              torch.tensor([track_id_1]).to(self.device),
                                              torch.tensor([track_id_2]).to(self.device),
                                              cur_target['track_ids'][index + 1:]))
        cur_target['boxes'] = torch.cat((cur_target['boxes'][:index],
                                          torch.cat((cur_target['boxes'][index, :4], torch.zeros(4).to(self.device)))[None],
                                          torch.cat((cur_target['boxes'][index, 4:], torch.zeros(4).to(self.device)))[None],
                                          cur_target['boxes'][index + 1:]))
        cur_target['labels'] = torch.cat((cur_target['labels'][:index],
                                           torch.tensor([0, 1]).long().to(self.device)[None],
                                           torch.tensor([0, 1]).long().to(self.device)[None],
                                           cur_target['labels'][index + 1:]))
        cur_target['flexible_divisions'] = torch.cat((cur_target['flexible_divisions'][:index],
                                                       torch.tensor([False]).to(self.device),
                                                       cur_target['flexible_divisions'][index:]))
        edge_1 = cur_target['is_touching_edge_orig'][cur_target['track_ids_orig'] == track_id_1_orig]
        edge_2 = cur_target['is_touching_edge_orig'][cur_target['track_ids_orig'] == track_id_2_orig]
        cur_target['is_touching_edge'] = torch.cat((cur_target['is_touching_edge'][:index], edge_1, edge_2,
                                                     cur_target['is_touching_edge'][index + 1:]))

        if 'masks' in cur_target:
            N, _, H, W = cur_target['masks'].shape
            cur_target['masks'] = torch.cat((
                cur_target['masks'][:index],
                torch.cat((cur_target['masks'][index, :1], torch.zeros((1, H, W)).to(self.device, dtype=torch.uint8)))[None],
                torch.cat((cur_target['masks'][index, 1:], torch.zeros((1, H, W)).to(self.device, dtype=torch.uint8)))[None],
                cur_target['masks'][index + 1:]))

    def get_random_indices(self, targets, target_name, prev_indices):
        rand_num = torch.rand(1)

        for target, prev_ind in zip(targets, prev_indices):
            prev_out_ind, prev_target_ind = prev_ind

            random_subset_mask_orig = torch.randperm(len(prev_target_ind))

            if target['main'][target_name]['empty'] or target_name == 'prev_target':
                target['main'][target_name]['prev_ind'] = [prev_ind[0][random_subset_mask_orig],
                                                           prev_ind[1][random_subset_mask_orig]]
                target['main'][target_name]['prev_ind_orig'] = [prev_ind[0][random_subset_mask_orig],
                                                                prev_ind[1][random_subset_mask_orig]]
                target['main'][target_name]['random_subset_mask'] = random_subset_mask_orig
                target['main'][target_name]['random_subset_mask_orig'] = random_subset_mask_orig
                continue

            if self.no_data_aug:
                random_subset_mask = random_subset_mask_orig
            elif rand_num > 0.5:
                random_subset_mask = random_subset_mask_orig[:len(prev_target_ind)]
            else:
                random_subset_mask = random_subset_mask_orig[:max(len(prev_target_ind) - torch.randint(0, max(len(prev_target_ind) // 5, 2), (1,)), 1)]

            target['main'][target_name]['prev_ind'] = [prev_out_ind[random_subset_mask],
                                                       prev_target_ind[random_subset_mask]]
            target['main'][target_name]['prev_ind_orig'] = [prev_out_ind[random_subset_mask_orig],
                                                            prev_target_ind[random_subset_mask_orig]]
            target['main'][target_name]['random_subset_mask'] = random_subset_mask
            target['main'][target_name]['random_subset_mask_orig'] = random_subset_mask_orig

            if (target_name == 'cur_target' and
                    torch.tensor([t['main']['cur_target']['empty'] for t in targets]).sum() == 0 and
                    torch.tensor([t['main']['prev_target']['empty'] for t in targets]).sum() == 0):

                if self.dn_track:
                    target['dn_track'] = {'cur_target': {}}
                    for k in self.copy_dict_keys:
                        if k in target['main']['cur_target']:
                            target['dn_track']['cur_target'][k] = target['main']['cur_target'][k].clone()
                    target['dn_track']['prev_target'] = target['main']['prev_target'].copy()
                    if 'fut_target' in target['main']:
                        target['dn_track']['fut_target'] = target['main']['fut_target'].copy()
                    target['dn_track']['training_method'] = 'dn_track'
                    target['dn_track']['cur_target']['prev_ind'] = [prev_out_ind[random_subset_mask_orig],
                                                                     prev_target_ind[random_subset_mask_orig]]
                    target['dn_track']['cur_target']['prev_ind_orig'] = [prev_out_ind[random_subset_mask_orig],
                                                                          prev_target_ind[random_subset_mask_orig]]
                    target['dn_track']['man_track'] = target['main']['man_track'].clone()

                if self.dn_track_group:
                    target['dn_track_group'] = {'cur_target': {}}
                    for k in self.copy_dict_keys:
                        if k in target['main']['cur_target']:
                            target['dn_track_group']['cur_target'][k] = target['main']['cur_target'][k].clone()
                    target['dn_track_group']['prev_target'] = target['main']['prev_target'].copy()
                    if 'fut_target' in target['main']:
                        target['dn_track_group']['fut_target'] = target['main']['fut_target'].copy()
                    target['dn_track_group']['training_method'] = 'dn_track_group'
                    target['dn_track_group']['cur_target']['prev_ind'] = [prev_out_ind[random_subset_mask],
                                                                           prev_target_ind[random_subset_mask]]
                    target['dn_track_group']['cur_target']['prev_ind_orig'] = [prev_out_ind[random_subset_mask_orig],
                                                                                prev_target_ind[random_subset_mask_orig]]
                    target['dn_track_group']['cur_target']['random_subset_mask'] = random_subset_mask
                    target['dn_track_group']['man_track'] = target['main']['man_track'].clone()

    def separate_divided_cells(self, target, training_method, target_name, boxes=None, masks=None):
        for prev_ind_out in target[training_method][target_name]['prev_ind'][0]:
            if (target[training_method][target_name]['prev_ind'][0] == prev_ind_out).sum() == 2:
                inds = torch.where(target[training_method][target_name]['prev_ind'][0] == prev_ind_out)[0]
                orig_inds = target[training_method][target_name]['random_subset_mask'][inds]
                assert len(orig_inds) == 2 and (orig_inds[0] == orig_inds[1] + 1 or orig_inds[0] + 1 == orig_inds[1])
                ind = inds[torch.argmax(orig_inds)]
                if boxes is not None:
                    boxes[ind, :4] = boxes[ind, 4:]
                else:
                    masks[ind, 0] = masks[ind, 1]
        if boxes is not None:
            return boxes
        else:
            return masks

    def remove_new_cells(self, target, training_method, target_name, target_ind_match_matrix, prev_track_ids):
        new_cell_indices = sorted((target_ind_match_matrix.sum(0) == 0).nonzero(), reverse=True)
        man_track = target[training_method]['man_track']
        output_target = target[training_method][target_name]
        output_target['new_cell_ids'] = []
        framenb = output_target['framenb']

        for new_cell_ind in new_cell_indices:
            track_id = output_target['track_ids'][new_cell_ind]

            if track_id in man_track[:, -1]:
                dau_cells = man_track[man_track[:, -1] == track_id, 0]
                if man_track[man_track[:, 0] == dau_cells[0], 1] == framenb:
                    output_target['new_cell_ids'].extend([dau_cells[0].item(), dau_cells[1].item()])
                    assert training_method == 'dn_track_group'
                else:
                    output_target['new_cell_ids'].append(output_target['track_ids'][new_cell_ind[0]].item())
            else:
                output_target['new_cell_ids'].append(output_target['track_ids'][new_cell_ind[0]].item())

            if new_cell_ind == output_target['boxes'].shape[0] - 1:
                for fld in ('boxes', 'labels', 'track_ids', 'flexible_divisions', 'is_touching_edge'):
                    output_target[fld] = output_target[fld][:new_cell_ind]
                if 'masks' in output_target:
                    output_target['masks'] = output_target['masks'][:new_cell_ind]
            else:
                for fld in ('boxes', 'labels', 'track_ids', 'flexible_divisions', 'is_touching_edge'):
                    output_target[fld] = torch.cat((output_target[fld][:new_cell_ind], output_target[fld][new_cell_ind + 1:]), axis=0)
                if 'masks' in output_target:
                    output_target['masks'] = torch.cat((output_target['masks'][:new_cell_ind], output_target['masks'][new_cell_ind + 1:]), axis=0)

        target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(output_target['track_ids'])
        output_target['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
        output_target['cells_leaving_mask'] = ~target_ind_match_matrix.any(dim=1)
        if training_method == 'main':
            output_target['cells_leaving_mask'] = torch.cat((output_target['cells_leaving_mask'],
                                                              torch.tensor([False] * output_target['num_FPs']).bool().to(self.device)))
        output_target['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

        return target

    # ------------------------------------------------------------------
    # Build track query inputs for the transformer from Instances state
    # ------------------------------------------------------------------

    def _build_track_query_targets(self, track_instances: Instances,
                                   targets, target_name, prev_target_name,
                                   prev_out, last_frame_tracked):
        """Convert MOTR Instances state to the transformer's expected target format.

        Populates:
          - track_query_hs_embeds  (content query)
          - track_query_boxes      (reference points / positional query)
          - track_queries_TP_mask, track_queries_mask, etc.
          - Division handling (separate_divided_cells)

        Also handles FP generation and DN-track.
        """
        # First do the standard DETRTrackingBase random-index and FP calculation
        # for legacy compatibility, but use Instances state as the source of truth
        # for track_query_hs_embeds and track_query_boxes.

        n_instances = len(track_instances)

        for i, target in enumerate(targets):
            cur_tgt = target['main'][target_name]

            # Build GT track ID → instance index mapping
            gt_track_ids = cur_tgt.get('track_ids', torch.tensor([]))
            prev_track_ids_gt = target['main'][prev_target_name].get('track_ids', torch.tensor([]))

            # Mask cells that have a match in the current frame
            tim = track_instances.obj_idxes.unsqueeze(1).eq(gt_track_ids.unsqueeze(0))  # [N_inst, N_gt]
            target_ind_matching = tim.any(dim=1)[:n_instances]
            track_query_match_ids = tim.nonzero()[:, 1][:n_instances] if tim.any() else torch.tensor([], dtype=torch.long, device=self.device)

            # Division-ahead GT
            if len(track_query_match_ids) > 0 and 'labels' in cur_tgt:
                cur_tgt['track_query_div_ahead_gt'] = (
                    cur_tgt['labels'][track_query_match_ids, 1] == 1
                ).float()
            else:
                cur_tgt['track_query_div_ahead_gt'] = torch.tensor([], device=self.device)

            # Handle FN divisions (same as DETRTrackingBase)
            target_ind_not_matched_idx = (tim.sum(dim=0) == 0).nonzero()[:, 0]
            if len(target_ind_not_matched_idx) > 0:
                count = 0
                for nidx in range(len(target_ind_not_matched_idx)):
                    target_ind_not_matched_i = target_ind_not_matched_idx[nidx] + count
                    if cur_tgt['boxes'][target_ind_not_matched_i][-1] > 0:
                        self.update_target(target['main'], target_name, target_ind_not_matched_i)
                        count += 1
                tim = track_instances.obj_idxes.unsqueeze(1).eq(cur_tgt['track_ids'].unsqueeze(0))
                target_ind_matching = tim.any(dim=1)[:n_instances]
                track_query_match_ids = tim.nonzero()[:, 1][:n_instances] if tim.any() else torch.tensor([], dtype=torch.long, device=self.device)

            # Use MOTR Instances as track queries
            boxes = track_instances.ref_pts.clone()
            hs = track_instances.output_embedding.clone()

            # Apply mask-based box initialisation if available
            if self.masks and self.init_boxes_from_masks and len(boxes) > 0 and 'pred_masks' in prev_out:
                prev_indices_main = target['main'][target_name].get('prev_ind')
                if prev_indices_main is not None:
                    masks_inst = prev_out['pred_masks'][i, prev_indices_main[0]]
                    if last_frame_tracked:
                        masks_inst = self.separate_divided_cells(target, 'main', target_name, masks=masks_inst)
                    masks_inst = masks_inst[:, 0]
                    masks_filt = torch.zeros_like(masks_inst)
                    argmax = torch.argmax(masks_inst, axis=0)
                    for m in range(masks_inst.shape[0]):
                        masks_filt[m, argmax == m] = masks_inst[m, argmax == m]
                    mask_boxes = box_ops.masks_to_boxes(masks_filt > 0, cxcywh=True)
                    # Only update boxes for tracked instances with valid prev_ind
                    boxes[:len(mask_boxes)] = mask_boxes

            # Velocity offset from GT (training only)
            if (self.training and target_name == 'cur_target' and len(boxes) > 0 and
                    'prev_prev_target' in target['main'] and
                    not target['main']['prev_prev_target'].get('empty', False)):
                pprev_target = target['main']['prev_prev_target']
                if 'track_ids' in pprev_target and len(pprev_target['track_ids']) > 0:
                    pprev_ids = pprev_target['track_ids']
                    gt_prev_centers = target['main'][prev_target_name]['boxes'][:, :2]
                    match_pprev = track_instances.obj_idxes.unsqueeze(1).eq(pprev_ids.unsqueeze(0))
                    has_pprev = match_pprev.any(1)
                    if has_pprev.any():
                        pprev_idx = match_pprev.float().argmax(1)
                        cur_ids = track_instances.obj_idxes
                        match_prev = cur_ids.unsqueeze(1).eq(prev_track_ids_gt.unsqueeze(0))
                        has_prev = match_prev.any(1)
                        prev_idx = match_prev.float().argmax(1)
                        gt_prev_c = target['main'][prev_target_name]['boxes'][:, :2]
                        gt_pprev_c = pprev_target['boxes'][:, :2]
                        vel = torch.zeros_like(boxes[:, :2])
                        valid = has_pprev & has_prev
                        if valid.any():
                            vel[valid] = (gt_prev_c[prev_idx[valid]] - gt_pprev_c[pprev_idx[valid]])
                            boxes[:, :2] = (boxes[:, :2] + vel).clamp(0.0, 1.0)

            # FP generation (standard from DETRTrackingBase)
            n_FPs = int(cur_tgt.get('num_FPs', 0))
            if n_FPs > 0:
                FP_boxes, FP_hs = self.get_FP_boxes(target['main'], target_name, i, prev_out)
                boxes = torch.cat((boxes, FP_boxes))
                hs = torch.cat((hs, FP_hs))

            # Populate target dict
            cur_tgt['track_query_hs_embeds'] = hs
            cur_tgt['track_query_boxes'] = boxes
            cur_tgt['target_ind_matching'] = torch.cat([
                target_ind_matching,
                torch.zeros(n_FPs, dtype=torch.bool, device=self.device)
            ])
            cur_tgt['cells_leaving_mask'] = ~cur_tgt['target_ind_matching']
            cur_tgt['track_query_match_ids'] = track_query_match_ids
            cur_tgt['track_queries_TP_mask'] = torch.cat([
                cur_tgt['target_ind_matching'],
                torch.zeros(self.num_queries, dtype=torch.bool, device=self.device)
            ])
            cur_tgt['track_queries_mask'] = torch.cat([
                torch.ones(n_instances + n_FPs, dtype=torch.bool, device=self.device),
                torch.zeros(self.num_queries, dtype=torch.bool, device=self.device)
            ])
            cur_tgt['track_queries_fal_pos_mask'] = torch.cat([
                ~cur_tgt['target_ind_matching'],
                torch.zeros(self.num_queries, dtype=torch.bool, device=self.device)
            ])
            cur_tgt['num_queries'] = n_instances + n_FPs + self.num_queries

    # ------------------------------------------------------------------
    # Main MOTR forward (replaces DETRTrackingBase.forward)
    # ------------------------------------------------------------------

    def forward(self, samples: utils.NestedTensor, targets: list = None):

        if not self.train_model:
            # Pure detection (no tracking)
            return super().forward(samples, targets)

        if targets is None:
            raise NotImplementedError

        track = torch.rand(1) > 0.1

        backprop_context = nullcontext if self._backprop_prev_frame else torch.no_grad

        if not track:
            # 10% of the time, run as a pure detector (no tracking) for stability
            with backprop_context():
                for target in targets:
                    target['main']['cur_target']['track_queries_mask'] = \
                        torch.zeros(self.num_queries, dtype=torch.bool, device=self.device)
            out, targets, features, memory, hs = super().forward(
                [t['cur_image'] for t in targets], targets, 'cur_target')

            if self.masks and self.init_boxes_from_masks:
                out['pred_masks'] = self.forward_prediction_heads(hs[-1], self.final_mask_embed_index)

            out['prev_outputs'] = None
            out['prev_prev_outputs'] = None
            out['training_methods'] = ['main']
            return out, targets, features, memory, hs

        # ---- Tracking forward ----
        with backprop_context():
            if max(target['main']['cur_target']['boxes'].shape[0] for target in targets) > self.num_queries:
                raise NotImplementedError("More GT cells than num_queries")

            track_two_frames = torch.rand(1) > 1 / 9
            self.last_frame_tracked = False

            # ================================================================
            # PREV PREV frame (detection only, no track queries)
            # ================================================================
            for target in targets:
                target['main']['prev_prev_target']['track_queries_mask'] = \
                    torch.zeros(self.num_queries, dtype=torch.bool, device=self.device)

            prev_prev_out, _, _, _, hs_pprev = super().forward(
                [t['prev_prev_image'] for t in targets])

            if self.masks and self.init_boxes_from_masks:
                prev_prev_out['pred_masks'] = self.forward_prediction_heads(
                    hs_pprev[-1], self.final_mask_embed_index)

            prev_prev_indices, targets = self._matcher(
                prev_prev_out, targets, 'main', 'prev_prev_target')

            for t, target in enumerate(targets):
                target['main']['prev_prev_target']['track_queries_mask'] = \
                    torch.zeros(self.num_queries, dtype=torch.bool, device=self.device)
                target['main']['prev_prev_target']['indices'] = prev_prev_indices[t]

            targets = utils.man_track_ids(targets, 'main', 'prev_prev_target', 'prev_target')

            # ================================================================
            # PREV frame – initialise Instances from prev_prev detections
            # ================================================================
            self.get_random_indices(targets, 'prev_target', prev_prev_indices)
            self.calc_num_FPs(targets, 'prev_target')

            # Init MOTR track instances from prev_prev detections
            # One Instances per batch element (we run batch_size=1 in practice)
            track_instances_list = []
            for i, target in enumerate(targets):
                ti = self._generate_empty_tracks()
                prev_ind = target['main']['prev_target']['prev_ind']

                # Initialise slots from prev_prev predictions
                n_tracked = len(prev_ind[0])
                if n_tracked > 0:
                    if self.masks and self.init_boxes_from_masks and 'pred_masks' in prev_prev_out:
                        masks = prev_prev_out['pred_masks'][i, prev_ind[0]]
                        masks = masks[:, 0]
                        masks_filt = torch.zeros_like(masks)
                        argmax = torch.argmax(masks, axis=0)
                        for m in range(masks.shape[0]):
                            masks_filt[m, argmax == m] = masks[m, argmax == m]
                        init_boxes = box_ops.masks_to_boxes(masks_filt > 0, cxcywh=True)
                    else:
                        init_boxes = prev_prev_out['pred_boxes'][i, prev_ind[0], :4].detach().clone()

                    init_hs = prev_prev_out['hs_embed'][i, prev_ind[0]].detach().clone()

                    # Assign GT track IDs from prev_target (target gt matches prev_prev detections)
                    gt_track_ids = target['main']['prev_prev_target']['track_ids']
                    for k, (out_idx, tgt_idx) in enumerate(zip(prev_ind[0], prev_ind[1])):
                        if tgt_idx < len(gt_track_ids) and k < len(ti.obj_idxes):
                            ti.obj_idxes[k] = gt_track_ids[tgt_idx]

                    ti.ref_pts[:n_tracked] = init_boxes[:n_tracked]
                    ti.output_embedding[:n_tracked] = init_hs[:n_tracked]

                track_instances_list.append(ti)

            # Build the prev_target target dict from Instances state
            for i, (ti, target) in enumerate(zip(track_instances_list, targets)):
                prev_tgt = target['main']['prev_target']
                prev_tgt['track_query_hs_embeds'] = ti.output_embedding
                prev_tgt['track_query_boxes'] = ti.ref_pts
                # Mark as prev frame tracking context
                prev_tgt['prev_prev_frame'] = True

                # Set standard track query masks
                n_ti = len(ti)
                prev_tgt['target_ind_matching'] = torch.zeros(n_ti, dtype=torch.bool, device=self.device)
                prev_tgt['track_queries_TP_mask'] = torch.zeros(n_ti + self.num_queries, dtype=torch.bool, device=self.device)
                prev_tgt['track_queries_mask'] = torch.cat([
                    torch.ones(n_ti, dtype=torch.bool, device=self.device),
                    torch.zeros(self.num_queries, dtype=torch.bool, device=self.device)
                ])
                prev_tgt['track_queries_fal_pos_mask'] = torch.cat([
                    torch.zeros(n_ti, dtype=torch.bool, device=self.device),
                    torch.zeros(self.num_queries, dtype=torch.bool, device=self.device)
                ])
                prev_tgt['num_FPs'] = 0
                prev_tgt['num_queries'] = n_ti + self.num_queries

        # PREV frame forward
        self.last_frame_tracked = True

        if self._backprop_prev_frame:
            prev_out, _, _, _, hs_prev = super().forward(
                [t['prev_image'] for t in targets], targets, 'prev_target')
        else:
            with torch.no_grad():
                prev_out, _, _, _, hs_prev = super().forward(
                    [t['prev_image'] for t in targets], targets, 'prev_target')

        if self.masks and self.init_boxes_from_masks:
            prev_out['pred_masks'] = self.forward_prediction_heads(
                hs_prev[-1], self.final_mask_embed_index)

        prev_indices, targets = self._matcher(prev_out, targets, 'main', 'prev_target')

        # ================================================================
        # Update Instances with prev frame predictions, then QIM + MemoryBank
        # ================================================================
        for i, (ti, target) in enumerate(zip(track_instances_list, targets)):
            prev_tgt = target['main']['prev_target']
            n_ti = len(ti)

            if not prev_tgt.get('empty', False):
                with torch.no_grad():
                    scores = prev_out['pred_logits'][i, :n_ti].sigmoid().max(dim=-1).values
                ti.scores = scores
                ti.pred_logits = prev_out['pred_logits'][i, :n_ti].detach()
                ti.pred_boxes = prev_out['pred_boxes'][i, :n_ti].detach()
                ti.output_embedding = prev_out['hs_embed'][i, :n_ti].detach()

                # Update matched GT IDs for prev frame
                if len(prev_indices[i][0]) > 0:
                    out_idx, tgt_idx = prev_indices[i]
                    gt_ids = target['main']['prev_target']['track_ids']
                    for k, (oi, ti_idx) in enumerate(zip(out_idx, tgt_idx)):
                        if oi < n_ti and ti_idx < len(gt_ids):
                            ti.obj_idxes[oi] = gt_ids[ti_idx]

            # MemoryBank + QIMv2 update
            with torch.no_grad():
                ti = self.memory_bank(ti)
                ti = self.qim(ti)

            track_instances_list[i] = ti

        if self.flex_div:
            targets = update_early_or_late_track_divisions(
                prev_out, targets, 'main', 'prev_prev_target', 'prev_target', 'cur_target')

        for t, target in enumerate(targets):
            target['main']['prev_target']['track_ids_track'] = target['main']['prev_target']['track_ids'].clone()

        # ================================================================
        # Prepare CUR frame targets using updated Instances
        # ================================================================
        self.get_random_indices(targets, 'cur_target', prev_indices)
        self.calc_num_FPs(targets, 'cur_target')

        for i, (ti, target) in enumerate(zip(track_instances_list, targets)):
            cur_tgt = target['main']['cur_target']

            # Use QIMv2-updated embeddings and ref_pts as track queries for cur frame
            n_ti = len(ti)
            n_FPs = int(cur_tgt.get('num_FPs', 0))

            # Handle FP generation (reuse prev_out predictions as noised FPs)
            boxes = ti.ref_pts.clone()
            hs = ti.output_embedding.clone()

            if n_FPs > 0:
                FP_boxes, FP_hs = self.get_FP_boxes(target['main'], 'cur_target', i, prev_out)
                boxes = torch.cat((boxes, FP_boxes))
                hs = torch.cat((hs, FP_hs))

            # MOTR-style: identify which Instances are TP for cur_target
            gt_track_ids_cur = cur_tgt.get('track_ids', torch.tensor([]))
            tim_cur = ti.obj_idxes.unsqueeze(1).eq(gt_track_ids_cur.unsqueeze(0)) if len(gt_track_ids_cur) > 0 else \
                torch.zeros(n_ti, 0, dtype=torch.bool, device=self.device)
            target_ind_matching = tim_cur.any(dim=1)

            # FN division box expansion (same as DETRTrackingBase)
            tgt_not_matched = (tim_cur.sum(dim=0) == 0).nonzero()[:, 0]
            if len(tgt_not_matched) > 0:
                count = 0
                for nidx in range(len(tgt_not_matched)):
                    tind = tgt_not_matched[nidx] + count
                    if cur_tgt['boxes'][tind][-1] > 0:
                        self.update_target(target['main'], 'cur_target', tind)
                        count += 1
                gt_track_ids_cur = cur_tgt['track_ids']
                tim_cur = ti.obj_idxes.unsqueeze(1).eq(gt_track_ids_cur.unsqueeze(0)) if len(gt_track_ids_cur) > 0 else \
                    torch.zeros(n_ti, 0, dtype=torch.bool, device=self.device)
                target_ind_matching = tim_cur.any(dim=1)

            track_query_match_ids = tim_cur.nonzero()[:, 1] if tim_cur.any() else \
                torch.tensor([], dtype=torch.long, device=self.device)

            # Division-ahead GT
            if len(track_query_match_ids) > 0 and 'labels' in cur_tgt:
                cur_tgt['track_query_div_ahead_gt'] = (
                    cur_tgt['labels'][track_query_match_ids, 1] == 1
                ).float()
            else:
                cur_tgt['track_query_div_ahead_gt'] = torch.tensor([], device=self.device)

            # Populate target dict for transformer and criterion
            cur_tgt['track_query_hs_embeds'] = hs
            cur_tgt['track_query_boxes'] = boxes
            cur_tgt['target_ind_matching'] = torch.cat([
                target_ind_matching,
                torch.zeros(n_FPs, dtype=torch.bool, device=self.device)
            ])
            cur_tgt['cells_leaving_mask'] = ~cur_tgt['target_ind_matching']
            cur_tgt['track_query_match_ids'] = track_query_match_ids
            cur_tgt['track_queries_TP_mask'] = torch.cat([
                cur_tgt['target_ind_matching'],
                torch.zeros(self.num_queries, dtype=torch.bool, device=self.device)
            ])
            cur_tgt['track_queries_mask'] = torch.cat([
                torch.ones(n_ti + n_FPs, dtype=torch.bool, device=self.device),
                torch.zeros(self.num_queries, dtype=torch.bool, device=self.device)
            ])
            cur_tgt['track_queries_fal_pos_mask'] = torch.cat([
                ~cur_tgt['target_ind_matching'],
                torch.zeros(self.num_queries, dtype=torch.bool, device=self.device)
            ])
            cur_tgt['num_queries'] = n_ti + n_FPs + self.num_queries

            # DN-track
            if 'dn_track' in target:
                dn_tgt = target['dn_track']['cur_target']
                dn_tgt_track_ids = dn_tgt.get('track_ids', torch.tensor([]))
                prev_track_ids_dn = target['main']['prev_target']['track_ids'][
                    dn_tgt['prev_ind'][1]] if len(dn_tgt.get('prev_ind', [[], []])[1]) > 0 else torch.tensor([])

                tim_dn = prev_track_ids_dn.unsqueeze(1).eq(dn_tgt_track_ids.unsqueeze(0)) if len(dn_tgt_track_ids) > 0 else \
                    torch.zeros(len(prev_track_ids_dn), 0, dtype=torch.bool, device=self.device)

                dn_tgt['target_ind_matching'] = torch.cat([
                    tim_dn.any(dim=1),
                    torch.zeros(dn_tgt.get('num_FPs', 0), dtype=torch.bool, device=self.device)
                ])
                dn_tgt['cells_leaving_mask'] = ~dn_tgt['target_ind_matching']
                dn_tgt['track_queries_TP_mask'] = dn_tgt['target_ind_matching']
                dn_tgt['track_queries_mask'] = torch.ones_like(dn_tgt['target_ind_matching']).bool()
                dn_tgt['track_queries_fal_pos_mask'] = ~dn_tgt['target_ind_matching']

                dn_boxes = target['main']['prev_target']['boxes'][dn_tgt['prev_ind'][1], :4].clone()
                dn_hs = self.dn_track_embedding.weight.repeat(len(dn_boxes), 1)
                dn_boxes = box_ops.add_noise_to_boxes(dn_boxes, self.dn_track_l1, self.dn_track_l2)

                if dn_tgt.get('num_FPs', 0) > 0:
                    dn_boxes = torch.cat((dn_boxes, torch.zeros((dn_tgt['num_FPs'], 4), device=self.device)))
                    dn_hs = torch.cat((dn_hs, torch.zeros((dn_tgt['num_FPs'], self.hidden_dim), device=self.device)))

                dn_tgt['track_query_boxes'] = dn_boxes
                dn_tgt['track_query_hs_embeds'] = dn_hs
                dn_tgt['num_queries'] = len(dn_tgt['track_queries_mask'])
                dn_tgt['track_query_match_ids'] = tim_dn.nonzero()[:, 1] if tim_dn.any() else \
                    torch.tensor([], dtype=torch.long, device=self.device)

            # DN-track-group (uses QIM-updated embeddings + noise)
            if 'dn_track_group' in target:
                dtg_tgt = target['dn_track_group']['cur_target']
                dtg_prev_ind = dtg_tgt['prev_ind']
                dtg_boxes = prev_out['pred_boxes'][i, dtg_prev_ind[0]].clone().detach()[:, :4]
                dtg_hs = prev_out['hs_embed'][i, dtg_prev_ind[0]].clone()

                if self.last_frame_tracked:
                    dtg_boxes = self.separate_divided_cells(target, 'dn_track_group', 'cur_target', boxes=dtg_boxes)

                dtg_hs = dtg_hs + torch.normal(0, self.tgt_noise, size=dtg_hs.shape, device=self.device)

                dtg_track_ids = dtg_tgt.get('track_ids', torch.tensor([]))
                dtg_prev_ids = target['main']['prev_target']['track_ids'][dtg_prev_ind[1]] if len(dtg_prev_ind[1]) > 0 else torch.tensor([])
                tim_dtg = dtg_prev_ids.unsqueeze(1).eq(dtg_track_ids.unsqueeze(0)) if len(dtg_track_ids) > 0 else \
                    torch.zeros(len(dtg_prev_ids), 0, dtype=torch.bool, device=self.device)

                n_dtg_FPs = int(dtg_tgt.get('num_FPs', 0))
                if n_dtg_FPs > 0:
                    dtg_boxes = torch.cat((dtg_boxes, torch.zeros((n_dtg_FPs, 4), device=self.device)))
                    dtg_hs = torch.cat((dtg_hs, torch.zeros((n_dtg_FPs, self.hidden_dim), device=self.device)))

                dtg_tgt['target_ind_matching'] = torch.cat([
                    tim_dtg.any(dim=1),
                    torch.zeros(n_dtg_FPs, dtype=torch.bool, device=self.device)
                ])
                dtg_tgt['cells_leaving_mask'] = ~dtg_tgt['target_ind_matching']
                dtg_tgt['track_queries_TP_mask'] = dtg_tgt['target_ind_matching']
                dtg_tgt['track_queries_mask'] = torch.ones_like(dtg_tgt['target_ind_matching']).bool()
                dtg_tgt['track_queries_fal_pos_mask'] = ~dtg_tgt['target_ind_matching']
                dtg_tgt['track_query_boxes'] = dtg_boxes
                dtg_tgt['track_query_hs_embeds'] = dtg_hs
                dtg_tgt['num_queries'] = len(dtg_tgt['track_queries_mask'])
                dtg_tgt['track_query_match_ids'] = tim_dtg.nonzero()[:, 1] if tim_dtg.any() else \
                    torch.tensor([], dtype=torch.long, device=self.device)

        # ================================================================
        # CUR frame forward (gradient flows here)
        # ================================================================
        out, targets, features, memory, hs = super().forward(
            [t['cur_image'] for t in targets], targets, 'cur_target')

        if self.masks and self.init_boxes_from_masks:
            out['pred_masks'] = self.forward_prediction_heads(hs[-1], self.final_mask_embed_index)

        # Update Instances with cur frame predictions
        for i, (ti, target) in enumerate(zip(track_instances_list, targets)):
            cur_tgt = target['main']['cur_target']
            n_ti = len(ti)
            if not cur_tgt.get('empty', False):
                ti.scores = out['pred_logits'][i, :n_ti].detach().sigmoid().max(dim=-1).values
                ti.pred_logits = out['pred_logits'][i, :n_ti].detach()
                ti.pred_boxes = out['pred_boxes'][i, :n_ti].detach()
                ti.output_embedding = out['hs_embed'][i, :n_ti].detach()

        # Store prev outputs for logging / auxiliary losses
        out['prev_outputs'] = prev_out
        out['prev_prev_outputs'] = prev_prev_out
        out['training_methods'] = ['main']
        if 'dn_track' in targets[0]:
            out['training_methods'].append('dn_track')
        if 'dn_track_group' in targets[0]:
            out['training_methods'].append('dn_track_group')

        return out, targets, features, memory, hs
