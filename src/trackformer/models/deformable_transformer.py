# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import math

import torch
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_
import torch.nn.functional as F

from ..util.misc import inverse_sigmoid
from ..util import box_ops
from .ops.modules import MSDeformAttn
from .transformer import _get_clones, _get_activation_fn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, num_queries=30, batch_size = 2,
                 use_dab=False, high_dim_query_update=False, no_sine_embed=False,
                 multi_frame_attention_separate_encoder=False, 
                 refine_track_queries=False, refine_div_track_queries=False,
                 init_enc_queries_embeddings=False,device='cuda',masks=False,
                 dn_enc_l1=0, dn_enc_l2=0, init_boxes_from_masks=False,
                 enc_masks=False,enc_FN=0,avg_attn_weight_maps=True,
                 tgt_noise=1e-6,use_img_for_mask=False, num_OD_layers=0,use_div_box_as_ref_pts=False,
                 use_qim=False, num_qim_layers=1,
                 temporal_dropout_prob=0.0, track_query_noise=0.0):
        super().__init__()

        self.temporal_dropout_prob = temporal_dropout_prob
        self.track_query_noise = track_query_noise
        self.d_model = d_model
        self.batch_size = batch_size
        self.nhead = nhead
        self.two_stage = two_stage
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.multi_frame_attention_separate_encoder = multi_frame_attention_separate_encoder
        self.use_dab = use_dab
        self.device = device
        self.enc_masks = enc_masks
        self.enc_FN = enc_FN
        self.avg_attn_weight_maps = avg_attn_weight_maps
        self.tgt_noise = tgt_noise
        self.use_img_for_mask = use_img_for_mask

        self.refine_track_queries = refine_track_queries
        self.refine_div_track_queries = refine_div_track_queries
        self.init_enc_queries_embeddings = init_enc_queries_embeddings
        self.init_boxes_from_masks = init_boxes_from_masks

        self.num_OD_layers = num_OD_layers

        if self.num_OD_layers > 1:
            raise NotImplementedError
        
        if self.refine_track_queries:
            self.track_embedding = nn.Embedding(1,self.d_model)

        self.use_qim = use_qim
        if self.use_qim:
            self.qim = QueryInteractionModule(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                n_levels=num_feature_levels,
                n_points=dec_n_points,
                num_layers=num_qim_layers,
            )


        if self.refine_div_track_queries:
            self.div_track_embedding = nn.Embedding(2,self.d_model)

        if self.init_enc_queries_embeddings:
            self.enc_query_embeddings = nn.Embedding(1,self.d_model)

        self.dn_enc_l1 = dn_enc_l1
        self.dn_enc_l2 = dn_enc_l2

        enc_num_feature_levels = num_feature_levels
        if multi_frame_attention_separate_encoder:
            enc_num_feature_levels = enc_num_feature_levels // 2

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, enc_num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points,avg_attn_weight_maps)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, use_dab=use_dab, d_model=d_model, return_intermediate_dec=return_intermediate_dec,
                                                    high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed, num_OD_layers=self.num_OD_layers,use_div_box_as_ref_pts=use_div_box_as_ref_pts)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            if self.use_dab:
                self.pos_trans = nn.Linear(d_model * 2, d_model)
                self.pos_trans_norm = nn.LayerNorm(d_model)
            else:
                self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
                self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if not self.use_dab:
                self.reference_points = nn.Linear(d_model, 2)

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"

        self._reset_parameters()

    def _reset_parameters(self):
        for name,p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        # num_pos_feats = 128
        num_pos_feats = 144
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=self.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        # base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=self.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=self.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_PEM(self,features,memory):

            offset = 0
            memory_slices = []
            batch_size, _, channels = memory.shape
            for height,width in self.spatial_shapes:
                memory_slice = memory[:, offset:offset + height * width].permute(0, 2, 1).view(
                    batch_size, channels, height, width)
                memory_slices.append(memory_slice)
                offset += height * width

            memory = memory_slices[::-1] # this will be used in segmentation; flip the order to follow DINO

            # Get mask_features needed for segmentation
            fpns = [features[i].tensors for i in range(self.num_feature_levels)][::-1]
            mask_size = fpns[-1].shape[-2:]

            mask_features = []

            if self.use_img_for_mask:
                images = self.samples.tensors.to(self.device)
                images_enc = self.img_encoder(images)
                mask_size = images.shape[-2:]
                # fpns.append(images_enc)

            for fidx, fpn in enumerate(fpns):
                
                cur_fpn = self.lateral_layers[fidx](fpn)
                y = cur_fpn + F.interpolate(memory[fidx], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False) # Mask DINO uses just the largest spatial shape output from encoder 
                # Below is a deterministic version of F.interpolate
                # y = cur_fpn + F.interpolate(memory[-1], size=cur_fpn.shape[-2:]) # Mask DINO uses just the largest spatial shape output from encoder
                y = self.output_layers[fidx](y)
                mask_features.append(F.interpolate(y, size=mask_size, mode="bilinear", align_corners=False))
                # Below is a deterministic version of F.interpolate
                # mask_features.append(F.interpolate(y, size=mask_size))

            if self.use_img_for_mask:
                mask_features.append(images_enc)

            mask_features = torch.cat(mask_features,1)
            self.all_mask_features = self.mask_features(mask_features)

    def forward(self, features, srcs, masks, pos_embeds, query_embed=None, targets=None, output_target=None, query_attn_mask=None,training_methods=[]):
        assert self.two_stage or query_embed is not None
        if not self.two_stage:
            assert torch.sum(torch.isnan(query_embed)) == 0, 'Nan in reference points'

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=self.device)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        if self.multi_frame_attention_separate_encoder and self.multi_frame_attention:
            level_start_index = torch.cat((spatial_shapes[:spatial_shapes.shape[0]//2].new_zeros((1, )), spatial_shapes[:spatial_shapes.shape[0]//2].prod(1).cumsum(0)[:-1]))
            prev_memory = self.encoder(
                src_flatten[:, :src_flatten.shape[1] // 2],
                spatial_shapes[:self.num_feature_levels // 2],
                valid_ratios[:, :self.num_feature_levels // 2],
                level_start_index,
                lvl_pos_embed_flatten[:, :src_flatten.shape[1] // 2],
                mask_flatten[:, :src_flatten.shape[1] // 2],
                )
            memory = self.encoder(
                src_flatten[:, src_flatten.shape[1] // 2:],
                spatial_shapes[self.num_feature_levels // 2:],
                valid_ratios[:, self.num_feature_levels // 2:],
                level_start_index,
                lvl_pos_embed_flatten[:, src_flatten.shape[1] // 2:],
                mask_flatten[:, src_flatten.shape[1] // 2:],
                )
            memory = torch.cat([memory, prev_memory], 1)
        else:
            level_start_index = torch.cat((spatial_shapes[:spatial_shapes.shape[0]].new_zeros((1, )), spatial_shapes[:spatial_shapes.shape[0]].prod(1).cumsum(0)[:-1]))
            memory = self.encoder(src_flatten, spatial_shapes, valid_ratios, level_start_index, lvl_pos_embed_flatten, mask_flatten)

        self.spatial_shapes = spatial_shapes # needed for detr_segmentation.py

        if self.masks:
            self.get_PEM(features,memory)

        enc_outputs = {}

        if self.two_stage:
            
            if self.multi_frame_attention_separate_encoder:
                output_memory, output_proposals = self.gen_encoder_output_proposals(memory[:,:memory.shape[1]//2], mask_flatten[:,:mask_flatten.shape[1]//2], spatial_shapes[:spatial_shapes.shape[0]//2])
            else:
                output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            # enc_outputs_class = self.class_embed(output_memory)[...,:1]
            enc_outputs_class = self.class_embed[-1](output_memory)[...,:1]
            enc_outputs_coord_unact = self.bbox_embed[-1](output_memory)[...,:output_proposals.shape[-1]] + output_proposals

            assert self.num_queries <= enc_outputs_class.shape[1]
            topk = self.num_queries

            # topk = min(self.two_stage_num_proposals,enc_outputs_class.shape[1])
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_undetach = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_detach = topk_coords_undetach.detach()
            reference_points = topk_coords_detach.sigmoid()
            topk_classes = torch.gather(enc_outputs_class, 1, topk_proposals.unsqueeze(-1))

            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))  # unsigmoid

            topk_classes = topk_classes[...,:1]
            enc_outputs['pred_logits'] = torch.cat((topk_classes,torch.zeros_like(topk_classes)),axis=-1)  # I use weight map to discard the second prediction in loss function
         
            enc_outputs['pred_boxes'] = torch.cat((topk_coords_undetach.sigmoid(),torch.zeros_like(topk_coords_undetach)),axis=-1)
            enc_outputs['topk_proposals'] = topk_proposals

            if self.multi_frame_attention_separate_encoder:
                enc_outputs['spatial_shapes'] = spatial_shapes[:spatial_shapes.shape[0]//2]
            else:
                enc_outputs['spatial_shapes'] = spatial_shapes

            if self.init_enc_queries_embeddings:
                tgt = self.enc_query_embeddings.weight.repeat(memory.shape[0],self.num_queries,1) # Don't use batch size here in case at end of epoch, only one sample is used
            else:
                # gather tgt
                tgt = tgt_undetach.detach()

            if self.enc_masks and self.masks:

                outputs_mask = self.forward_prediction_heads(tgt_undetach,-1)
                enc_outputs['pred_masks'] = outputs_mask
                
                # if self.init_boxes_from_masks:
                if self.init_boxes_from_masks: 
                    flatten_mask = outputs_mask.detach().flatten(0, 1)[:,0]

                    #TODO need to make this work for torch instead of numpy so everything can stay on the gpu
                    # h, w = outputs_mask.shape[-2:]
                    # refpoint_embed = box_ops.masks_to_boxes(flatten_mask > 0)
                    # refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed) / torch.as_tensor([w, h, w, h],dtype=torch.float,device=self.device)
                    
                    refpoint_embed = box_ops.masks_to_boxes(flatten_mask > 0,cxcywh=True) # bbox are 0 to 1 so we don't have to resize masks before getting bboxes
                    refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4)
                    enc_outputs['mask_enc_boxes'] = refpoint_embed.clone().cpu()
                    reference_points = refpoint_embed

            if query_embed is not None: # used for dn_object or if you want to add extra object queries on top of two_stage
                reference_points = torch.cat((reference_points,query_embed[..., self.d_model:].sigmoid()),axis=1)
                tgt = torch.cat((tgt,query_embed[..., :self.d_model]),axis=1)

                query_embed = None

        elif self.use_dab:
            reference_points = query_embed[..., self.d_model:].sigmoid() 
            tgt = query_embed[..., :self.d_model]
            tgt_oqs_clone = tgt[:,-self.num_queries:].clone()
            boxes_oqs_clone = reference_points[:,-self.num_queries:].clone()
            query_embed = None
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

            reference_points = self.reference_points(query_embed).sigmoid()

        num_track_queries = 0

        if targets is not None and 'track_query_hs_embeds' in targets[0]['main'][output_target]:

            prev_hs_embed = torch.stack([t['main'][output_target]['track_query_hs_embeds'] for t in targets])
            prev_boxes = torch.stack([t['main'][output_target]['track_query_boxes'] for t in targets])

            num_track_queries += prev_boxes.shape[1]

            if self.refine_track_queries:
                prev_hs_embed += self.track_embedding.weight

            # Temporal dropout: randomly zero individual track query embeddings during
            # training. Forces the model to detect/track without always relying on temporal
            # context, improving generalization to occlusions and track starts.
            if self.training and self.temporal_dropout_prob > 0.0 and prev_hs_embed.shape[1] > 0:
                keep = torch.bernoulli(
                    torch.ones(prev_hs_embed.shape[0], prev_hs_embed.shape[1], 1, device=self.device)
                    * (1.0 - self.temporal_dropout_prob)
                )
                prev_hs_embed = prev_hs_embed * keep

            # Track query noise: additive Gaussian noise on track embeddings during training.
            # Simulates imperfect temporal signal (occlusion, identity confusion) at inference.
            if self.training and self.track_query_noise > 0.0 and prev_hs_embed.shape[1] > 0:
                prev_hs_embed = prev_hs_embed + torch.randn_like(prev_hs_embed) * self.track_query_noise

            if self.use_qim and prev_hs_embed.shape[1] > 0:
                prev_hs_embed = self.qim(
                    prev_hs_embed, prev_boxes[..., :4],
                    memory, spatial_shapes, valid_ratios, level_start_index, mask_flatten,
                )

            if not self.use_dab:
                prev_query_embed = torch.zeros_like(prev_hs_embed)
                query_embed = torch.cat([prev_query_embed, query_embed], dim=1)

            prev_tgt = prev_hs_embed
            tgt = torch.cat([prev_tgt, tgt], dim=1)

            reference_points = torch.cat([prev_boxes[..., :reference_points.shape[-1]], reference_points], dim=1)

            if 'dn_track' in targets[0]:
                
                num_dn_track = targets[0]['dn_track']['cur_target']['num_queries']
                assert num_dn_track > 0

                num_total_queries = query_attn_mask.shape[0]

                for target in targets:
                    target['dn_track']['start_query_ind'] = num_total_queries
                    target['dn_track']['end_query_ind'] = num_total_queries + num_dn_track

                prev_hs_embed_dn_track = torch.stack([t['dn_track'][output_target]['track_query_hs_embeds'] for t in targets])
                prev_boxes_dn_track = torch.stack([t['dn_track'][output_target]['track_query_boxes'] for t in targets])

                if not self.use_dab:
                    prev_query_embed_dn_track = torch.zeros_like(prev_hs_embed_dn_track)
                    query_embed = torch.cat([query_embed,prev_query_embed_dn_track], dim=1)

                prev_tgt_dn_track = prev_hs_embed_dn_track

                tgt = torch.cat([tgt,prev_tgt_dn_track], dim=1)
                reference_points = torch.cat([reference_points,prev_boxes_dn_track[..., :reference_points.shape[-1]]], dim=1)

                new_query_attn_mask = torch.zeros((tgt.shape[1],tgt.shape[1]),device=self.device).bool()
                new_query_attn_mask[:query_attn_mask.shape[0],:query_attn_mask.shape[1]] = query_attn_mask

                new_query_attn_mask[:-num_dn_track,-num_dn_track:] = True
                new_query_attn_mask[-num_dn_track:,:-num_dn_track] = True
                # new_query_attn_mask[-num_dn_track:,self.num_queries:-num_dn_track] = True

                query_attn_mask = new_query_attn_mask

                training_methods.append('dn_track')

                if self.dn_track_group and 'dn_track_group' in targets[0]:

                    num_dn_track_group = targets[0]['dn_track_group'][output_target]['num_queries']
                    assert num_dn_track_group > 0

                    for target in targets:
                        target['dn_track_group']['start_query_ind'] = query_attn_mask.shape[0]
                        target['dn_track_group']['end_query_ind'] = query_attn_mask.shape[0] + num_dn_track_group

                    prev_hs_embed_dn_track_group = torch.stack([t['dn_track_group'][output_target]['track_query_hs_embeds'] for t in targets])
                    prev_boxes_dn_track_group = torch.stack([t['dn_track_group'][output_target]['track_query_boxes'] for t in targets])

                    if self.use_qim and prev_hs_embed_dn_track_group.shape[1] > 0:
                        prev_hs_embed_dn_track_group = self.qim(
                            prev_hs_embed_dn_track_group, prev_boxes_dn_track_group[..., :4],
                            memory, spatial_shapes, valid_ratios, level_start_index, mask_flatten,
                        )


                    if not self.use_dab:
                        prev_hs_embed_dn_track_group = torch.zeros_like(prev_hs_embed_dn_track_group)
                        query_embed = torch.cat([query_embed,prev_boxes_dn_track_group], dim=1)

                    prev_tgt_dn_track_group = prev_hs_embed_dn_track_group

                    tgt = torch.cat([tgt,prev_tgt_dn_track_group], dim=1)
                    reference_points = torch.cat([reference_points,prev_boxes_dn_track_group[..., :reference_points.shape[-1]]], dim=1)

                    new_query_attn_mask = torch.zeros((tgt.shape[1],tgt.shape[1]),device=self.device).bool()
                    new_query_attn_mask[:query_attn_mask.shape[0],:query_attn_mask.shape[1]] = query_attn_mask

                    new_query_attn_mask[:-num_dn_track_group,-num_dn_track_group:] = True
                    # new_query_attn_mask[-num_dn_track_group:,:-num_dn_track_group] = True
                    new_query_attn_mask[-num_dn_track_group:,self.num_queries + prev_boxes.shape[1] :-num_dn_track_group] = True
                    new_query_attn_mask[-num_dn_track_group:,:prev_boxes.shape[1]] = True

                    query_attn_mask = new_query_attn_mask

                    training_methods.append('dn_track_group')

        OD_outputs = None

        if targets is not None and 'prev_prev_frame' in targets[0]['main'][output_target]: # dn_track queies are mixed with object queries; easiest solution to make the dn_track queries not see the object queries because i only care about the dn_track queries
            query_attn_mask[:-self.num_queries,-self.num_queries:] = True
            query_attn_mask[-self.num_queries:,:-self.num_queries] = True # object queries are needed to detect any new objects

        if self.num_OD_layers > 0:
            OD_tgt = tgt[:,num_track_queries:num_track_queries + self.num_queries]
            OD_query_attn_mask = torch.zeros((OD_tgt.shape[1],OD_tgt.shape[1]),device=self.device).bool()
            OD_reference_points = reference_points[:,num_track_queries:num_track_queries + self.num_queries]
            OD_hs, OD_inter_references_points, OD_outputs_class, OD_outputs_bbox = self.decoder.forward_OD_only(OD_tgt, OD_reference_points, memory, spatial_shapes, valid_ratios, level_start_index, query_embed, mask_flatten, OD_query_attn_mask)

            OD_outputs = {'pred_logits': OD_outputs_class[-1],
                        'pred_boxes': OD_outputs_bbox[-1],
                        'hs_embed': OD_hs[-1],
                        'training_methods': 'OD_only',}
            
            assert (tgt[:,num_track_queries:num_track_queries + self.num_queries] == OD_tgt).all()
            reference_points[:,num_track_queries:num_track_queries + self.num_queries] = OD_inter_references_points.clone()
          
        init_reference_point = reference_points[None]

        # decoder
        hs, inter_references_points, outputs_class, outputs_bbox = self.decoder( tgt, reference_points, memory, spatial_shapes, valid_ratios, 
                                                          level_start_index, query_embed, mask_flatten, query_attn_mask)

        reference_points = torch.cat((init_reference_point,inter_references_points),axis=0)     

        return hs, memory, reference_points, outputs_class, outputs_bbox, enc_outputs, training_methods, OD_outputs

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes,level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, valid_ratios, level_start_index, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,avg_attn_weight_maps=True):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.avg_attn_weight_maps = avg_attn_weight_maps

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, src_level_start_index, src_padding_mask=None, query_attn_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=query_attn_mask, average_attn_weights=self.avg_attn_weight_maps)[0].transpose(0, 1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, src_level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, use_dab=False, d_model=256, return_intermediate_dec=False, high_dim_query_update=False, no_sine_embed=False,num_OD_layers=0,use_div_box_as_ref_pts=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers - num_OD_layers)
        self.num_layers = num_layers
        self.return_intermediate_dec = return_intermediate_dec
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.mask_embed = None
        self.use_div_box_as_ref_pts = use_div_box_as_ref_pts

        #### DAB-DETR
        self.use_dab = use_dab
        self.d_model = d_model
        self.no_sine_embed = no_sine_embed
        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            if self.no_sine_embed:
                self.ref_point_head = MLP(4, d_model, d_model, 3)
            else:
                self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 2)
        #### DAB-DETR

        self.num_OD_layers = num_OD_layers

        if self.num_OD_layers > 0:
            self.OD_layers = _get_clones(decoder_layer,self.num_OD_layers)

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_valid_ratios, src_level_start_index,
                query_pos=None, src_padding_mask=None, query_attn_mask=None):
        output = tgt

        #### DAB-DETR
        if self.use_dab:
            assert query_pos is None
        #### DAB-DETR

        intermediate = []
        intermediate_reference_points = []
        intermediate_cls = []
        intermediate_bboxes = []
        bboxes = reference_points.clone()

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            #### DAB-DETR
            if self.use_dab:
                if self.no_sine_embed:
                    raw_query_pos = self.ref_point_head(reference_points_input)
                else:
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :],self.d_model) # bs, nq, d_model * 2
                    raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            if self.high_dim_query_update and lid != 0:
                query_pos = query_pos + self.high_dim_query_proj(output) 
            #### DAB-DETR

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, query_attn_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:

                # cls = self.class_embed(output)
                cls = self.class_embed[lid](output)
                layer_output = self.bbox_embed[lid](output) 

                if self.use_div_box_as_ref_pts:
                    inverse_box = inverse_sigmoid(bboxes)

                    if lid == 0:
                        bboxes = torch.cat((layer_output[:,:,:4] + inverse_box, layer_output[:,:,4:] + inverse_box),axis=-1).sigmoid()
                    else:
                        bboxes = (layer_output + inverse_box).sigmoid()

                    div = cls[:,:,1].sigmoid() > 0.5

                    new_reference_points = torch.zeros((bboxes.shape[0],bboxes.shape[1],4),device=bboxes.device)
                    new_reference_points[~div] = bboxes[~div][:,:4].detach()
                  
                    if div.sum() > 0:
                        div_bboxes = bboxes[div].detach()
                        combined_boxes = box_ops.combine_boxes_parallel(div_bboxes[:,:4],div_bboxes[:,4:])

                        new_reference_points[div] = combined_boxes

                        reference_points = new_reference_points.detach()
                                    
                else:
                    
                    inverse_sig_ref_pts =  inverse_sigmoid(reference_points)
                    reference_points = (layer_output[:,:,:4] + inverse_sig_ref_pts).sigmoid().detach()
                    bboxes = torch.cat((layer_output[:,:,:4] + inverse_sig_ref_pts, layer_output[:,:,4:] + inverse_sig_ref_pts),axis=-1).sigmoid()

            if self.return_intermediate_dec:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_cls.append(cls)
                intermediate_bboxes.append(bboxes)

        if self.return_intermediate_dec:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(intermediate_cls), torch.stack(intermediate_bboxes)
        
        return output[None], reference_points[None], cls[None], bboxes[None]

    def forward_OD_only(self, tgt, reference_points, src, src_spatial_shapes, src_valid_ratios, src_level_start_index,
                query_pos=None, src_padding_mask=None, query_attn_mask=None):
        
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        intermediate_cls = []
        intermediate_bboxes = []

        for lid, layer in enumerate(self.OD_layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            #### DAB-DETR
            if self.use_dab:
                if self.no_sine_embed:
                    raw_query_pos = self.ref_point_head(reference_points_input)
                else:
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :],self.d_model) # bs, nq, d_model * 2
                    raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            if self.high_dim_query_update and lid != 0:
                query_pos = query_pos + self.high_dim_query_proj(output) 
            #### DAB-DETR

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, query_attn_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:

                cls = self.class_embed[-2](output)
                layer_output = self.bbox_embed[-2](output)                                   

                inverse_sig_ref_pts =  inverse_sigmoid(reference_points)
                reference_points = (layer_output[:,:,:4] + inverse_sig_ref_pts).sigmoid().detach()
                bboxes = torch.cat((layer_output[:,:,:4] + inverse_sig_ref_pts, layer_output[:,:,4:] + inverse_sig_ref_pts),axis=-1).sigmoid()

            if self.return_intermediate_dec:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_cls.append(cls)
                intermediate_bboxes.append(bboxes)

        if self.return_intermediate_dec:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(intermediate_cls), torch.stack(intermediate_bboxes)
        
        return output[None], reference_points[None], cls[None], bboxes[None]

class QueryInteractionModule(nn.Module):
    """MOTR-style Query Interaction Module (QIM). Disabled by default (use_qim=False)."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout, n_levels, n_points, num_layers):
        super().__init__()
        layer = QueryInteractionLayer(d_model, nhead, dim_feedforward, dropout, n_levels, n_points)
        self.layers = _get_clones(layer, num_layers)

    def forward(self, tgt, reference_points, memory, spatial_shapes, valid_ratios, level_start_index, memory_padding_mask=None):
        output = tgt
        for layer in self.layers:
            if reference_points.shape[-1] == 4:
                ref_pts = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                ref_pts = reference_points[:, :, None] * valid_ratios[:, None]
            output = layer(output, ref_pts, memory, spatial_shapes, level_start_index, memory_padding_mask)
        return output


class QueryInteractionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, n_levels, n_points):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformAttn(d_model, n_levels, nhead, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, reference_points, memory, spatial_shapes, level_start_index, memory_padding_mask=None):
        # self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # cross-attention with encoder memory
        tgt2 = self.cross_attn(tgt, reference_points, memory, spatial_shapes, level_start_index, memory_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # ffn
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MLP(nn.Module):
    """
        Adapted from DAB-DETR
        Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gen_sineembed_for_position(pos_tensor,d_model):
    '''Adapted from DAB-DETR'''
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 288)
    scale = 2 * math.pi
    dim_t = torch.arange(torch.div(d_model,2,rounding_mode='floor'), dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t,2,rounding_mode='floor') / torch.div(d_model,2,rounding_mode='floor'))
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos
