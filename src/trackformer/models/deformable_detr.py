# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import copy
import torch
import torch.nn.functional as F
from torch import nn

from ..util import box_ops
from ..util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list, MLP


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR():
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, tracking, backbone, num_classes, num_queries, num_feature_levels,device,
                 aux_loss=True, with_box_refine=False, two_stage=False, overflow_boxes=False,
                 multi_frame_attention=False, use_dab=True, random_refpoints_xy=False, dn_object=False,
                 dn_object_FPs=False, dn_object_l1 = 0, dn_object_l2 = 0, refine_object_queries=False,
                 share_bbox_layers=True,use_img_for_mask=False,masks=False,freeze_backbone=False,
                 freeze_backbone_and_encoder=False, use_temporal_memory=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO,
                         we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        
        self.tracking = tracking
        self.masks = masks
        self.device = device
        self.num_queries = num_queries
        self.hidden_dim = self.d_model

        # Temporal Track Memory: per-dimension learnable gate that blends the current
        # decoder output with the previous track embedding. This gives each track a
        # persistent representation whose update rate is learned per feature dimension.
        # Gate initialized to sigmoid(3) ≈ 0.95 so training starts near the current
        # behavior (mostly trust current frame) and gradually learns temporal mixing.
        self.use_temporal_memory = use_temporal_memory
        if self.use_temporal_memory:
            self.temporal_gate = nn.Parameter(torch.ones(self.hidden_dim) * 3.0)
        self.overflow_boxes = overflow_boxes
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 2)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 8, 3)
        self.div_ahead_embed = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.div_ahead_embed.weight)
        nn.init.zeros_(self.div_ahead_embed.bias)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

        # match interface with deformable DETR
        self.input_proj = nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)

        self.backbone = backbone
        self.aux_loss = aux_loss

        self.use_img_for_mask = use_img_for_mask

        # super().__init__(backbone, transformer, num_classes, num_queries, device, two_stage, aux_loss)
        self.multi_frame_attention = multi_frame_attention
        self.overflow_boxes = overflow_boxes
        self.num_feature_levels = num_feature_levels
        self.num_queries = num_queries
        self.share_bbox_layers = share_bbox_layers


        self.dn_object = dn_object

        if self.dn_object:
            self.dn_object_FPs = dn_object_FPs
            self.dn_object_l1 = dn_object_l1
            self.dn_object_l2 = dn_object_l2
            self.dn_object_embedding = nn.Embedding(1,self.hidden_dim)

        self.use_dab = use_dab
        self.random_refpoints_xy = random_refpoints_xy

        self.refine_object_queries = refine_object_queries

        if self.refine_object_queries:
            self.object_embedding = nn.Embedding(1,self.hidden_dim)

        ### DAB-DETR
        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, self.hidden_dim*2)
            else:
                self.tgt_embed = nn.Embedding(num_queries, self.hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries, 4)
                if random_refpoints_xy:
                    self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False
        ### DAB-DETR

        num_channels = backbone.num_channels[-num_feature_levels:]
        
        if num_feature_levels > 1:
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            num_backbone_outs = len(backbone.strides) - 1

            input_proj_list = []
            for i in range(num_feature_levels):
                in_channels = num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(self.hidden_dim // 8, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(self.hidden_dim // 2, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(self.hidden_dim, self.hidden_dim),
                )])
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = self.decoder.num_layers

        if two_stage:
            num_pred += 1

        if with_box_refine:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0.)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0.)
            if self.share_bbox_layers:
                self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
                self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            else:
                self.class_embed = _get_clones(self.class_embed, num_pred)    
                self.bbox_embed = _get_clones(self.bbox_embed, num_pred)    

            # hack implementation for iterative bounding box refinement
            self.decoder.bbox_embed = self.bbox_embed
            self.decoder.class_embed = self.class_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0.)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0.)

            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.decoder.bbox_embed = None
            self.decoder.class_embed = None

        if freeze_backbone_and_encoder:
            for name, param in self.named_parameters():
                if 'decoder' not in name:
                    param.requires_grad_(False)

        if freeze_backbone:
            for name, param in self.named_parameters():
                if 'backbone' in name:
                    param.requires_grad_(False)  

    def forward(self, samples: NestedTensor, targets: list = None, target_name: str = 'cur_target'):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
            
        features, pos = self.backbone(samples)

        self.samples = samples

        features_all = features

        features = features[-self.num_feature_levels:]

        src_list = []
        mask_list = [] # mask for image size smaller than target size
        pos_list = [] # pos embeddings
        frame_features = [features]

        for frame, frame_feat in enumerate(frame_features):
            if self.multi_frame_attention:
                pos_list.extend([p[:, frame] for p in pos[-self.num_feature_levels:]])
            else:
                pos_list.extend(pos[-self.num_feature_levels:])

            for l, feat in enumerate(frame_feat):

                src, mask = feat.decompose()
                src_list.append(self.input_proj[l](src))
                mask_list.append(mask)

                assert mask is not None

            if self.num_feature_levels > len(frame_feat):
                _len_srcs = len(frame_feat)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](frame_feat[-1].tensors)
                    else:
                        src = self.input_proj[l](src_list[-1])

                    _, m = frame_feat[0].decompose()
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]

                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    src_list.append(src)
                    mask_list.append(mask)
                    if self.multi_frame_attention:
                        pos_list.append(pos_l[:, frame])
                    else:
                        pos_list.append(pos_l)

        bs = src.shape[0]
        training_methods = ['main']
        #### DAB-DETR
        query_attn_mask = None 

        if self.use_dab:
            if self.two_stage:
                query_embeds = None
            else:
                tgt_embed = self.tgt_embed.weight.repeat(bs,1,1)      # nq, 256
                if self.refine_object_queries:
                    tgt_embed += self.object_embedding.weight
                refanchor = self.refpoint_embed.weight.repeat(bs,1,1)      # nq, 4
                query_embeds = torch.cat((tgt_embed, refanchor), dim=-1)

            num_queries = self.num_queries

            # Initialize the attn_mask
            if targets is not None and len(targets[0]) > 0 and 'track_query_hs_embeds' in targets[0]['main'][target_name]:
                num_track_queries = targets[0]['main'][target_name]['track_query_hs_embeds'].shape[0]
            else: 
                num_track_queries = 0

            num_total_queries = num_queries + num_track_queries 
            query_attn_mask = torch.zeros((num_total_queries,num_total_queries)).bool().to(self.device)

            if targets is not None and target_name == 'cur_target':
                for target in targets:
                    target['main']['start_query_ind'] = 0
                    target['main']['end_query_ind'] = num_total_queries

            #### DN-DETR for noised object detection
            if self.dn_object and target_name == 'cur_target' and targets is not None and torch.tensor([target['main'][target_name]['empty'] for target in targets]).sum() == 0: # If there is an empty chamber, skip all denoising
                training_methods.append('dn_object')

                num_boxes = torch.tensor([len(target['main'][target_name]['labels_orig']) - int(target['main'][target_name]['empty']) for target in targets]).to(self.device)
                num_FPs = max(num_boxes) - num_boxes

                if self.dn_object_FPs:
                    if num_FPs.max() < 2:
                        num_FPs += torch.randint(4,(1,)).to(self.device)

                num_dn_object_queries = max(num_boxes + num_FPs)

                if self.two_stage:
                    query_embed_dn_object = torch.zeros((bs,num_dn_object_queries,self.hidden_dim + 4)).to(self.device)
                else:
                    query_embed_dn_object = torch.zeros((bs,num_dn_object_queries,query_embeds.shape[-1])).to(self.device)

                for t,target in enumerate(targets):

                    random_mask = torch.randperm(target['main'][target_name]['boxes_orig'].shape[0]).to(self.device)

                    target['dn_object'] = {'training_method': 'dn_object', 'cur_target': {}}

                    target['dn_object']['man_track'] = target['main']['man_track'].clone()
                    target['dn_object']['cur_target']['framenb'] = target['main'][target_name]['framenb']
                    target['dn_object']['cur_target']['empty'] = target['main'][target_name]['empty']
                    target['dn_object']['cur_target']['track_queries_fal_pos_mask'] = torch.zeros((num_dn_object_queries)).bool().to(self.device)

                    target['dn_object']['prev_target'] = target['main']['prev_target'].copy()
                    target['dn_object']['fut_target'] = target['main']['fut_target'].copy()

                    dict_keys = ['boxes','labels','track_ids','flexible_divisions','is_touching_edge']

                    if self.masks:
                        dict_keys += ['masks']
                    for dict_key in dict_keys: # ['labels','boxes','masks','track_ids','empty']
                        for orig in ['','_orig']:
                            if dict_key in target['main'][target_name]:
                                target['dn_object']['cur_target'][dict_key + orig] = target['main'][target_name][dict_key + '_orig'].clone()[random_mask]

                    target['dn_object']['cur_target']['track_query_match_ids'] = torch.arange(target['dn_object']['cur_target']['boxes'].shape[0],dtype=torch.int64).to(self.device)

                    l_1 = self.dn_object_l1
                    l_2 = self.dn_object_l2

                    noised_boxes = box_ops.add_noise_to_boxes(target['dn_object']['cur_target']['boxes'][:,:4].clone(),l_1,l_2)

                    if num_FPs[t] > 0:
                        if self.dn_object_FPs:
                            random_FP_mask = torch.randperm(min(num_FPs[t],target['dn_object']['cur_target']['boxes'].shape[0]))
                            FP_boxes = target['dn_object']['cur_target']['boxes'][random_FP_mask,:4].clone()

                            if num_FPs[t] > FP_boxes.shape[0]: # Only add one FP per cell in image, otherwise, just add empty boxes. don't want to overload with 100 boxes for one cell if there is a big mismatch between two images in a batch
                                empty_boxes = torch.zeros((num_FPs[t].item() - FP_boxes.shape[0],4),dtype=FP_boxes.dtype).to(self.device)
                                FP_boxes = torch.cat((FP_boxes,empty_boxes),0)
                        else:
                            FP_boxes = torch.zeros((num_FPs[t].item() - FP_boxes.shape[0],4),dtype=FP_boxes.dtype).to(self.device)

                        FP_boxes = box_ops.add_noise_to_boxes(FP_boxes,l_1*4,l_2*4)
                        noised_boxes = torch.cat((noised_boxes, FP_boxes),axis=0)
                        # No tracking is done here; just a formality so it works in the matcher.py code; but there are FPs as in empty tracking boxes
                        target['dn_object']['cur_target']['track_queries_fal_pos_mask'][-num_FPs[t]:] = True

                    target['dn_object']['cur_target']['num_FPs'] = num_FPs[t]
                    
                    # Also a formality so it works in the mathcer.py code
                    target['dn_object']['cur_target']['track_queries_mask'] = torch.ones((num_dn_object_queries)).bool().to(self.device)
                    target['dn_object']['cur_target']['num_queries'] = num_dn_object_queries
                    target['dn_object']['cur_target']['noised_boxes'] = noised_boxes
                    target['dn_object']['start_query_ind'] = num_total_queries
                    target['dn_object']['end_query_ind'] = num_total_queries + num_dn_object_queries.item()

                    label_embedding = self.dn_object_embedding.weight.repeat(num_dn_object_queries,1)
                    query_embed_dn_object[t,:,:self.hidden_dim] = label_embedding
                    query_embed_dn_object[t,:,self.hidden_dim:] = noised_boxes      

                if self.two_stage:
                    query_embeds = query_embed_dn_object
                else:
                    query_embeds = torch.cat((query_embeds,query_embed_dn_object),axis=1)

                num_total_queries += num_dn_object_queries.item()
                new_query_attn_mask = torch.zeros((num_total_queries,num_total_queries)).bool().to(self.device)    
                new_query_attn_mask[:-num_dn_object_queries,:-num_dn_object_queries] = query_attn_mask
                new_query_attn_mask[-num_dn_object_queries:,:-num_dn_object_queries] = True
                new_query_attn_mask[:-num_dn_object_queries,-num_dn_object_queries:] = True
                query_attn_mask = new_query_attn_mask

        else:
            if self.two_stage:
                raise NotImplementedError
            else:
                query_embeds = self.query_embed.weight

        hs, memory, reference_points, outputs_class, outputs_bbox, enc_outputs, training_methods, OD_outputs = \
            super().forward(features_all, src_list, mask_list, pos_list, query_embeds, targets, target_name, query_attn_mask, training_methods)

        # Temporal Track Memory Gate: blend current decoder output with the stored
        # previous track embedding. Each feature dimension has a learned mixing ratio.
        # alpha[d] near 1 → trust current frame more (fast-changing features like position).
        # alpha[d] near 0 → trust history more (slow-changing identity features).
        hs_embed = hs[-1]
        if self.use_temporal_memory and num_track_queries > 0 and targets is not None:
            alpha = torch.sigmoid(self.temporal_gate)  # [D], values in (0,1)
            hs_embed = hs[-1].clone()
            for b, target in enumerate(targets):
                t_dict = target.get('main', target)
                t_cur = t_dict.get(target_name, t_dict)
                if 'track_query_hs_embeds' in t_cur:
                    prev_h = t_cur['track_query_hs_embeds']       # [N_t, D] - stored state
                    curr_h = hs[-1][b, :prev_h.shape[0]]          # [N_t, D] - decoder output
                    hs_embed[b, :prev_h.shape[0]] = alpha * curr_h + (1.0 - alpha) * prev_h

        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_bbox[-1],
               'hs_embed': hs_embed,
               'pred_div_ahead': self.div_ahead_embed(hs[-1]).squeeze(-1),
               'references': reference_points,
               'training_methods': training_methods}

        if OD_outputs:
               out['OD'] = OD_outputs

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_bbox, outputs_class)

        if bool(enc_outputs):
            out['two_stage'] = enc_outputs

        offset = 0
        memory_slices = []
        batch_size, _, channels = memory.shape
        for src in src_list:
            _, _, height, width = src.shape
            memory_slice = memory[:, offset:offset + height * width].permute(0, 2, 1).view(
                batch_size, channels, height, width)
            memory_slices.append(memory_slice)
            offset += height * width

        memory = memory_slices[::-1] # We only care about encoder output from the first frame; this will be used in segmentation; flip the order to follow DINO

        return out, targets, features_all, memory, hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord, outputs_class=None, outputs_mask=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        assert outputs_coord is not None and outputs_class is not None
        return [{'pred_boxes': a, 'pred_logits': b} for a, b in zip(outputs_coord[:-1], outputs_class[:-1])]