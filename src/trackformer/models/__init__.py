# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from .backbone import build_backbone
from .deformable_detr import DeformableDETR
from .detr import SetCriterion
from .detr_segmentation import (DeformableDETRSegm, DeformableDETRSegmTracking)
from .detr_tracking import DeformableDETRTracking
from .matcher import build_matcher


def build_model(args):
    num_classes = 1
    device = torch.device(args.device)
    backbone = build_backbone(args)
    matcher = build_matcher(args)

    detr_kwargs = {
        'tracking': args.tracking,
        'backbone': backbone,
        'num_classes': num_classes - 1 if args.focal_loss else num_classes,
        'num_queries': args.num_queries,
        'aux_loss': args.aux_loss,
        'overflow_boxes': args.overflow_boxes,
        'device': device,
        'use_dab': args.use_dab,
        'dn_object': args.dn_object,
        'dn_object_FPs': args.dn_object_FPs,
        'dn_object_l1': args.dn_object_l1,
        'dn_object_l2': args.dn_object_l2,
        'refine_object_queries': args.refine_object_queries,
        'share_bbox_layers': args.share_bbox_layers,
        'with_box_refine': args.with_box_refine,
        'num_queries': args.num_queries,
        'num_feature_levels': args.num_feature_levels,
        'two_stage': args.two_stage,
        'multi_frame_attention': args.multi_frame_attention,
        'use_img_for_mask': args.use_img_for_mask,
        'masks': args.masks,
        'freeze_backbone': args.freeze_backbone,
        'freeze_backbone_and_encoder': args.freeze_backbone_and_encoder,}
    
    tracking_kwargs = {
        'matcher': matcher,
        'backprop_prev_frame': args.backprop_prev_frame,
        'dn_track': args.dn_track,
        'dn_track_l1': args.dn_track_l1,
        'dn_track_l2': args.dn_track_l2,
        'dn_enc':args.dn_enc,
        'refine_div_track_queries': args.refine_div_track_queries,
        'flex_div': args.flex_div,
        'use_prev_prev_frame': args.use_prev_prev_frame,
        'num_queries': args.num_queries,
        'dn_track_group': args.dn_track_group,
        'tgt_noise': args.tgt_noise}

    mask_kwargs = {
        'freeze_detr': args.freeze_detr,
        'return_intermediate_masks': args.return_intermediate_masks,
        'mask_dim': args.mask_dim,}

    if args.deformable:
        args.feature_channels = backbone.num_channels

        num_feature_levels = args.num_feature_levels
        if args.multi_frame_attention:
            num_feature_levels *= 2

        transformer_kwargs = {
            'd_model': args.hidden_dim,
            'num_queries': args.num_queries,
            'num_feature_levels': args.num_feature_levels,
            'two_stage': args.two_stage,
            'nhead': args.nheads,
            'num_encoder_layers': args.enc_layers,
            'num_decoder_layers': args.dec_layers,
            'dim_feedforward': args.dim_feedforward,
            'dropout': args.dropout,
            'activation': "relu",
            'num_feature_levels': num_feature_levels,
            'dec_n_points': args.dec_n_points,
            'enc_n_points': args.enc_n_points,
            'two_stage': args.two_stage,
            'num_queries': args.num_queries,
            'batch_size': args.batch_size,
            'use_dab': args.use_dab,
            'return_intermediate_dec': True,
            'multi_frame_attention_separate_encoder': args.multi_frame_attention and args.multi_frame_attention_separate_encoder,
            'init_enc_queries_embeddings': args.init_enc_queries_embeddings,
            'dn_enc_l1': args.dn_enc_l1,
            'dn_enc_l2': args.dn_enc_l2,
            'init_boxes_from_masks': args.init_boxes_from_masks,
            'enc_masks': args.enc_masks,
            'enc_FN': args.enc_FN,
            'avg_attn_weight_maps': args.avg_attn_weight_maps,
            'use_img_for_mask': args.use_img_for_mask,
            'num_OD_layers': args.num_OD_layers,
            'use_div_box_as_ref_pts': args.use_div_box_as_ref_pts,
            'use_qim': args.use_qim,
            'num_qim_layers': args.num_qim_layers,}

        if args.tracking:
            if args.masks:
                model = DeformableDETRSegmTracking(mask_kwargs, tracking_kwargs, detr_kwargs,transformer_kwargs)
            else:
                model = DeformableDETRTracking(tracking_kwargs, detr_kwargs, transformer_kwargs)
        else:
            if args.masks:
                model = DeformableDETRSegm(mask_kwargs, detr_kwargs, transformer_kwargs)
            else:
                model = DeformableDETR(detr_kwargs, transformer_kwargs)
    else:
        raise NotImplementedError

    weight_dict = {'main_loss_ce': args.bbox_loss_coef,
                   'main_loss_bbox': args.bbox_loss_coef,
                   'main_loss_giou': args.giou_loss_coef,}

    if args.masks:
        weight_dict["main_loss_mask"] = args.mask_loss_coef
        weight_dict["main_loss_dice"] = args.dice_loss_coef

    training_methods = []
    if args.dn_track:
        training_methods.append('dn_track')
    if args.dn_track_group:
        training_methods.append('dn_track_group')
    if args.dn_object:
        training_methods.append('dn_object')
    if args.dn_enc:
        training_methods.append('dn_enc')
    if args.CoMOT:
        training_methods.append('CoMOT')


    weight_dict_TM = {}

    for weight_dict_key in list(weight_dict.keys()):
        for training_method in training_methods:
            if training_method == 'CoMOT':
                continue

            weight_dict_TM.update({f'{training_method}_{weight_dict_key.replace("main_","")}': weight_dict[weight_dict_key]})

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers-1-args.num_OD_layers):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

            for training_method in training_methods:
                aux_weight_dict.update({f'{training_method}_{k.replace("main_","")}_{i}': v for k, v in weight_dict.items()})

            if args.CoMOT:
                aux_weight_dict.update({f'CoMOT_{k.replace("main_","")}_{i}': v for k, v in weight_dict.items()})

        if args.two_stage:
            aux_weight_dict.update({f'two_stage_{k.replace("main_","")}': v for k, v in weight_dict.items()})

        if args.two_stage:
            aux_weight_dict.update({f'OD_{k.replace("main_","")}': v for k, v in weight_dict.items()})

        weight_dict.update(aux_weight_dict)

    weight_dict.update(weight_dict_TM)

    weight_dict.update({'loss': args.loss_coef})

    losses = ['labels', 'boxes']
    if args.masks:
        losses.append('masks')


    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tracking=args.tracking,
        args=args,)
    criterion.to(device)

    return model, criterion
