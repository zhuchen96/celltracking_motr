"""MOTR-style tracking with MemoryBank multi-frame temporal attention.

This module provides a parallel implementation to the existing DETRTrackingBase.
Enable it via `use_motr: true` in the config.  All existing functionality
(DN-track, DN-track-group, div_ahead, CoMOT, QIM, temporal-gate) is preserved
since MOTRTrackingBase inherits from DETRTrackingBase.

The MemoryBank (SelfMOTR-enhanced):
  • Each active track query cross-attends to its K most recent decoder output
    embeddings from previous frames.
  • mem_padding_mask [B, N, K] prevents empty bank slots from corrupting attention.
  • save_thresh: only update a bank slot at inference when detection score > thresh.
  • with_spatial_attn: optional spatial self-attention among current-frame tracks.

SelfMOTR dual-query:
  • use_dual_query: separate detection-only forward pass reusing cached encoder memory.
  • Provides a cleaner detection loss signal decoupled from tracking queries.
"""

import copy
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from ..util import misc as utils
from .deformable_detr import DeformableDETR
from .deformable_transformer import DeformableTransformer
from .detr_segmentation import DETRSegmBase
from .detr_tracking import DETRTrackingBase
from .matcher import HungarianMatcher
from .memory_bank import MemoryBank


class MOTRTrackingBase(DETRTrackingBase):
    """Extends DETRTrackingBase with MemoryBank multi-frame temporal attention.

    Only two public interfaces change relative to DETRTrackingBase:
      • __init__ gains `memory_bank_len`, `memory_bank_feedforward_dim`,
        `memory_bank_dropout` kwargs.
      • At inference, the target dict may contain `track_memory_bank`
        (shape [N, K, D]) which is applied before calling super().forward().
    """

    def __init__(self,
                 memory_bank_len: int = 3,
                 memory_bank_feedforward_dim: int = None,
                 memory_bank_dropout: float = 0.1,
                 memory_bank_save_thresh: float = 0.0,
                 memory_bank_spatial_attn: bool = False,
                 use_dual_query: bool = False,
                 num_queries_detect: int = 100,
                 detect_score_thresh: float = 0.5,
                 lambda_detect: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.memory_bank_len = memory_bank_len
        self.memory_bank_save_thresh = memory_bank_save_thresh
        self.use_dual_query = use_dual_query
        self.num_queries_detect = num_queries_detect
        self.detect_score_thresh = detect_score_thresh
        self.lambda_detect = lambda_detect
        # hidden_dim is set by DeformableDETR.__init__ (called before this)
        ffn_dim = memory_bank_feedforward_dim or self.hidden_dim * 4
        self.memory_bank_module = MemoryBank(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=ffn_dim,
            dropout=memory_bank_dropout,
            memory_len=memory_bank_len,
            save_thresh=memory_bank_save_thresh,
            with_spatial_attn=memory_bank_spatial_attn,
        )
        if use_dual_query:
            # Learnable embeddings for the detection-only auxiliary pass (SelfMOTR dual-query).
            # query_embed_detect: content embeddings [Q, D]
            # position_detect: raw reference points [Q, 4], applied .sigmoid() before use
            self.query_embed_detect = nn.Embedding(num_queries_detect, self.hidden_dim)
            self.position_detect = nn.Embedding(num_queries_detect, 4)
            nn.init.uniform_(self.position_detect.weight.data, 0, 1)

            # Dedicated class and box prediction heads for the detect-self pass.
            # These are separate from the shared tracking heads so the detect loss
            # does not pull the tracking heads toward "fire on all cells" behaviour.
            # One head per decoder layer (decoder.num_layers, not counting OD layers).
            n_dec = self.decoder.num_layers
            self.detect_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed[0]) for _ in range(n_dec)])
            self.detect_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed[0]) for _ in range(n_dec)])

    # ------------------------------------------------------------------
    # Training: build memory bank and enrich track queries
    # ------------------------------------------------------------------

    def add_track_queries_to_targets(self, targets, target_name, prev_target_name, prev_out):
        """Call parent then apply MemoryBank to cur_target track hs embeds."""
        super().add_track_queries_to_targets(targets, target_name, prev_target_name, prev_out)

        if target_name != 'cur_target':
            return

        for i, target in enumerate(targets):
            t_cur = target['main'][target_name]
            if 'track_query_hs_embeds' not in t_cur:
                continue
            hs = t_cur['track_query_hs_embeds']
            if hs.shape[0] == 0:
                continue

            bank, mask = self._build_memory_bank_for_target(target, i, hs.shape[0], hs.shape[1])
            if bank is None:
                continue

            # mem_padding_mask: [1, N, K] — True = empty slot
            mem_pad = mask.unsqueeze(0) if mask is not None else None
            hs_out = self.memory_bank_module(
                hs.unsqueeze(0), bank.unsqueeze(0), mem_padding_mask=mem_pad
            ).squeeze(0)
            t_cur['track_query_hs_embeds'] = hs_out

    def _build_memory_bank_for_target(self, target, batch_idx, N, D):
        """Return (bank [N, K, D], mask [N, K]) from prev_prev frame embeddings, or (None, None).

        mask[i, k] = True means slot k for track i is empty (padding).
        Uses `_motr_prev_prev_hs` temporarily stored in target['main'].
        """
        pprev_hs_all = target['main'].get('_motr_prev_prev_hs')
        if pprev_hs_all is None:
            return None, None

        pprev_target = target['main'].get('prev_prev_target')
        if pprev_target is None:
            return None, None

        pprev_indices = pprev_target.get('indices')
        if pprev_indices is None:
            return None, None
        pprev_pred_idx, pprev_gt_idx = pprev_indices
        if len(pprev_pred_idx) == 0:
            return None, None

        pprev_ids = pprev_target.get('track_ids')
        if pprev_ids is None or len(pprev_ids) == 0:
            return None, None

        prev_target = target['main']['prev_target']
        cur_target = target['main']['cur_target']
        prev_ind = cur_target['prev_ind']
        if len(prev_ind[1]) == 0:
            return None, None

        pprev_hs = pprev_hs_all[batch_idx]  # [total_queries, D]
        dev = pprev_hs.device

        # Matcher indices live on CPU; move everything to the model device
        pprev_pred_idx = pprev_pred_idx.to(dev)
        pprev_gt_idx = pprev_gt_idx.to(dev)
        pprev_ids = pprev_ids.to(dev)
        track_gt_ids = prev_target['track_ids'][prev_ind[1]].to(dev)

        K = self.memory_bank_len
        bank = torch.zeros(N, K, D, device=dev)
        mask = torch.ones(N, K, dtype=torch.bool, device=dev)  # True = empty

        for j in range(min(len(track_gt_ids), N)):
            gt_id = track_gt_ids[j]
            pprev_pos = (pprev_ids == gt_id).nonzero(as_tuple=False)
            if len(pprev_pos) == 0:
                continue
            pprev_gt_pos = pprev_pos[0, 0]
            pred_match = (pprev_gt_idx == pprev_gt_pos).nonzero(as_tuple=False)
            if len(pred_match) == 0:
                continue
            pprev_pred_pos = pprev_pred_idx[pred_match[0, 0]]
            if pprev_pred_pos < pprev_hs.shape[0]:
                bank[j, 0] = pprev_hs[pprev_pred_pos].detach()
                mask[j, 0] = False  # slot 0 is valid

        return bank, mask  # [N, K, D], [N, K]

    # ------------------------------------------------------------------
    # Inference: apply memory bank from tracker-maintained history
    # ------------------------------------------------------------------

    def _apply_memory_bank_inference(self, targets):
        """Apply MemoryBank to inference target dicts.

        Handles both flat dicts (from Tracker.step) and the pipeline's nested
        structure where track embeds live under target['main']['cur_target'].
        `track_memory_bank_mask` [N, K] bool tensor (True=empty) is optional.
        """
        for target in targets:
            # Resolve to the dict that holds track_query_hs_embeds
            if 'track_query_hs_embeds' in target:
                cur = target
            elif 'main' in target and 'track_query_hs_embeds' in target['main'].get('cur_target', {}):
                cur = target['main']['cur_target']
            else:
                continue
            hs = cur['track_query_hs_embeds']
            bank = cur.get('track_memory_bank')
            if bank is None or hs.shape[0] == 0:
                continue
            mask = cur.get('track_memory_bank_mask')  # [N, K] bool or None
            mem_pad = mask.unsqueeze(0) if mask is not None else None
            hs_out = self.memory_bank_module(
                hs.unsqueeze(0), bank.unsqueeze(0), mem_padding_mask=mem_pad
            ).squeeze(0)
            cur['track_query_hs_embeds'] = hs_out

    # ------------------------------------------------------------------
    # Dual-query: detection-only pass reusing cached encoder memory
    # ------------------------------------------------------------------

    def _run_detect_pass(self, B: int):
        """Run a detection-only decoder pass using the cached encoder memory.

        Uses dedicated detect_class_embed / detect_bbox_embed heads (not the shared
        tracking heads) so the detect-self loss does not affect tracking predictions.

        Returns a dict with pred_logits and pred_boxes (and aux_outputs if aux_loss),
        or None if not enabled or encoder cache is absent.
        """
        if not self.use_dual_query:
            return None
        if not hasattr(self, 'last_encoder_memory') or self.last_encoder_memory is None:
            return None

        tgt = self.query_embed_detect.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        ref_pts = self.position_detect.weight.sigmoid().unsqueeze(0).expand(B, -1, -1)  # [B, Q, 4]

        # Decoder-only pass; hs: [L, B, Q, D] — one entry per decoder layer.
        # outputs_class / outputs_bbox from the decoder use shared tracking heads;
        # we discard them and apply dedicated detect heads to the raw hs instead.
        hs, _, _ = self.decode_from_cache(tgt, ref_pts)

        # Apply dedicated heads per decoder layer
        det_cls  = [self.detect_class_embed[lid](hs[lid])  for lid in range(len(hs))]
        det_bbox = [self.detect_bbox_embed[lid](hs[lid]).sigmoid() for lid in range(len(hs))]

        detect_out = {
            'pred_logits': det_cls[-1],
            'pred_boxes':  det_bbox[-1],
        }
        if self.aux_loss:
            detect_out['aux_outputs'] = [
                {'pred_logits': det_cls[i], 'pred_boxes': det_bbox[i]}
                for i in range(len(hs) - 1)
            ]
        return detect_out

    # ------------------------------------------------------------------
    # Forward: inject prev_prev_hs for memory bank, handle inference
    # ------------------------------------------------------------------

    def forward(self, samples: utils.NestedTensor, targets: list = None):
        # ---- Inference (train_model=False) ----
        if not self.train_model:
            if targets is not None:
                self._apply_memory_bank_inference(targets)
            return super().forward(samples, targets)

        # ---- Training (train_model=True) ----
        # We replicate DETRTrackingBase.forward here so we can inject
        # prev_prev_hs into target['main'] before add_track_queries_to_targets
        # is called for cur_target.  Changes vs parent are marked with # MOTR.

        track = torch.rand(1) > 0.1

        if targets is None:
            raise NotImplementedError

        backprop_context = torch.no_grad
        if self._backprop_prev_frame:
            backprop_context = nullcontext

        if track:
            with backprop_context():

                if max([target['main']['cur_target']['boxes'].shape[0]
                        for target in targets]) > self.num_queries:
                    raise NotImplementedError

                track_two_frames = torch.rand(1) > 1 / 9

                if track_two_frames:
                    # PREV PREV
                    self.last_frame_tracked = False

                    prev_prev_out, _, _, _, hs = super(DETRTrackingBase, self).forward(
                        [t['prev_prev_image'] for t in targets])
                    del _

                    if self.masks and self.init_boxes_from_masks:
                        prev_prev_out['pred_masks'] = self.forward_prediction_heads(
                            hs[-1], self.final_mask_embed_index)

                    prev_prev_indices, targets = self._matcher(
                        prev_prev_out, targets, 'main', 'prev_prev_target')

                    for t, target in enumerate(targets):
                        target['main']['prev_prev_target']['track_queries_mask'] = \
                            torch.zeros((self.num_queries)).bool().to(self.device)
                        target['main']['prev_prev_target']['indices'] = prev_prev_indices[t]

                    targets = utils.man_track_ids(
                        targets, 'main', 'prev_prev_target', 'prev_target')

                    self.get_random_indices(targets, 'prev_target', prev_prev_indices)
                    self.add_track_queries_to_targets(
                        targets, 'prev_target', 'prev_prev_target', prev_prev_out)

                    # PREV
                    prev_out, _, _, _, hs = super(DETRTrackingBase, self).forward(
                        [t['prev_image'] for t in targets], targets, 'prev_target')
                    del _

                    if self.masks and self.init_boxes_from_masks:
                        prev_out['pred_masks'] = self.forward_prediction_heads(
                            hs[-1], self.final_mask_embed_index)

                    self.last_frame_tracked = True

                    prev_indices, targets = self._matcher(
                        prev_out, targets, 'main', 'prev_target')

                    if self.flex_div:
                        from ..util.flex_div import update_early_or_late_track_divisions
                        targets = update_early_or_late_track_divisions(
                            prev_out, targets, 'main',
                            'prev_prev_target', 'prev_target', 'cur_target')

                    for t, target in enumerate(targets):
                        target['main']['prev_target']['track_ids_track'] = \
                            target['main']['prev_target']['track_ids'].clone()

                    new_prev_indices = []
                    for bi, (target, (prev_ind_out, prev_ind_tgt)) in enumerate(
                            zip(targets, prev_indices)):
                        if target['main']['prev_target']['empty']:
                            new_prev_indices.append((prev_ind_out, prev_ind_tgt))
                        else:
                            boxes = target['main']['prev_target']['boxes'][prev_ind_tgt]
                            boxes_orig = target['main']['prev_target']['boxes_orig']

                            new_prev_ind_tgt = torch.ones(
                                (len(boxes_orig),), dtype=torch.int64) * -1
                            new_prev_ind_out = torch.ones(
                                (len(boxes_orig),), dtype=torch.int64) * -1

                            count = 0
                            for b, box in enumerate(boxes):
                                if box[-1] > 0:
                                    new_prev_ind_out[b + count] = prev_ind_out[b]
                                    new_prev_ind_out[b + count + 1] = prev_ind_out[b]
                                    new_ind_1 = boxes_orig[:, :4].eq(box[:4]).all(-1).nonzero()[0][0]
                                    new_ind_2 = boxes_orig[:, :4].eq(box[4:]).all(-1).nonzero()[0][0]

                                    # 2×2 Hungarian: assign each pred half to the
                                    # geometrically closer daughter GT box.
                                    # Slot b+count always uses pred[:4] (lower orig
                                    # index, kept by separate_divided_cells).
                                    # Slot b+count+1 always uses pred[4:] (higher
                                    # orig index, swapped by separate_divided_cells).
                                    pred_box = prev_out['pred_boxes'][bi, prev_ind_out[b]].detach()
                                    gt_dau1 = boxes_orig[new_ind_1, :4]
                                    gt_dau2 = boxes_orig[new_ind_2, :4]
                                    cost_2x2 = np.array([
                                        [(pred_box[:4] - gt_dau1).abs().sum().item(),
                                         (pred_box[:4] - gt_dau2).abs().sum().item()],
                                        [(pred_box[4:] - gt_dau1).abs().sum().item(),
                                         (pred_box[4:] - gt_dau2).abs().sum().item()],
                                    ])
                                    _, col_ind = linear_sum_assignment(cost_2x2)
                                    # col_ind[0]: which daughter pred[:4] is closest to
                                    if col_ind[0] == 0:
                                        new_prev_ind_tgt[b + count] = new_ind_1
                                        new_prev_ind_tgt[b + count + 1] = new_ind_2
                                    else:
                                        new_prev_ind_tgt[b + count] = new_ind_2
                                        new_prev_ind_tgt[b + count + 1] = new_ind_1

                                    count += 1
                                else:
                                    new_ind = boxes_orig[:, :4].eq(box[:4]).all(-1).nonzero()[0][0]
                                    new_prev_ind_tgt[b + count] = new_ind
                                    new_prev_ind_out[b + count] = prev_ind_out[b]

                            assert -1 not in new_prev_ind_out and -1 not in new_prev_ind_tgt
                            new_prev_indices.append((new_prev_ind_out, new_prev_ind_tgt))

                    prev_indices = new_prev_indices

                    for t, target in enumerate(targets):
                        target['main']['prev_target']['updated_indices'] = prev_indices[t]

                else:
                    prev_prev_out = None
                    prev_out, _, _, _, hs = super(DETRTrackingBase, self).forward(
                        [t['prev_image'] for t in targets])
                    del _

                    self.last_frame_tracked = False

                    if self.masks and self.init_boxes_from_masks:
                        prev_out['pred_masks'] = self.forward_prediction_heads(
                            hs[-1], self.final_mask_embed_index)

                    prev_indices, targets = self._matcher(
                        prev_out, targets, 'main', 'prev_target')

                    if self.flex_div:
                        from ..util.flex_div import update_object_detection
                        targets, prev_indices = update_object_detection(
                            prev_out, targets, prev_indices, self.num_queries,
                            'main', 'prev_prev_target', 'prev_target', 'cur_target')

                    for t, target in enumerate(targets):
                        target['main']['prev_target']['track_queries_mask'] = \
                            torch.zeros((self.num_queries)).bool().to(self.device)
                        target['main']['prev_target']['indices'] = prev_indices[t]

                targets = utils.man_track_ids(targets, 'main', 'prev_target', 'cur_target')

                self.get_random_indices(targets, 'cur_target', prev_indices)

                # MOTR: store prev_prev hs so add_track_queries_to_targets can build the bank
                if prev_prev_out is not None:
                    for target in targets:
                        target['main']['_motr_prev_prev_hs'] = prev_prev_out['hs_embed']

                self.add_track_queries_to_targets(
                    targets, 'cur_target', 'prev_target', prev_out)

                # MOTR: clean up temporary storage
                for target in targets:
                    target['main'].pop('_motr_prev_prev_hs', None)

                for target in targets:
                    target['track'] = True

        else:
            prev_prev_out = None
            prev_out = None

            for target in targets:
                target['main']['cur_target']['track_queries_mask'] = \
                    torch.zeros((self.num_queries)).bool().to(self.device)
                target['main']['cur_target']['track_queries_TP_mask'] = \
                    torch.zeros((self.num_queries)).bool().to(self.device)
                target['main']['cur_target']['num_queries'] = self.num_queries
                target['track'] = False

        out, targets, features, memory, hs = super(DETRTrackingBase, self).forward(
            samples, targets, 'cur_target')

        out['prev_outputs'] = prev_out
        out['prev_prev_outputs'] = prev_prev_out

        # Dual-query: detection-only auxiliary pass using cached encoder memory
        if self.use_dual_query:
            B = samples.tensors.shape[0] if hasattr(samples, 'tensors') else len(samples)
            out['detect_self_out'] = self._run_detect_pass(B)

        return out, targets, features, memory, hs


# ---------------------------------------------------------------------------
# Concrete classes (parallel to DeformableDETRTracking / DeformableDETRSegmTracking)
# ---------------------------------------------------------------------------

class MOTRDeformableDETRTracking(MOTRTrackingBase, DeformableDETR, DeformableTransformer):
    def __init__(self, tracking_kwargs, detr_kwargs, transformer_kwargs):
        DeformableTransformer.__init__(self, **transformer_kwargs)
        DeformableDETR.__init__(self, **detr_kwargs)
        MOTRTrackingBase.__init__(self, **tracking_kwargs)


class MOTRDeformableDETRSegmTracking(DETRSegmBase, MOTRTrackingBase, DeformableDETR, DeformableTransformer):
    def __init__(self, mask_kwargs, tracking_kwargs, detr_kwargs, transformer_kwargs):
        DeformableTransformer.__init__(self, **transformer_kwargs)
        DeformableDETR.__init__(self, **detr_kwargs)
        MOTRTrackingBase.__init__(self, **tracking_kwargs)
        DETRSegmBase.__init__(self, **mask_kwargs)
