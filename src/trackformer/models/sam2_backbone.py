"""SAM2 Hiera-L backbone for Cell-TRACTR."""

from typing import List
import torch
import torch.nn.functional as F
from torch import nn

from ..util.misc import NestedTensor
from .position_encoding import build_position_encoding


class SAM2HieraBackbone(nn.Module):
    """SAM2 Hiera-L image encoder as a detection backbone.

    Returns 4 feature levels at strides [4, 8, 16, 32], all projected to
    256 channels by the SAM2 neck convolutions.

    stage_ends = [1, 7, 43, 47]:
      stage 0 → blocks  0-1  → stride  4, 144 ch
      stage 1 → blocks  2-7  → stride  8, 288 ch
      stage 2 → blocks  8-43 → stride 16, 576 ch
      stage 3 → blocks 44-47 → stride 32, 1152 ch

    neck_convs (reversed order in SAM2):
      convs[0]: 1152 → 256  (stage 3)
      convs[1]:  576 → 256  (stage 2)
      convs[2]:  288 → 256  (stage 1)
      convs[3]:  144 → 256  (stage 0)
    """

    def __init__(self, checkpoint_path: str, unfreeze_stages: int = 1):
        super().__init__()
        from sam2.build_sam import build_sam2

        sam2 = build_sam2(
            'configs/sam2.1/sam2.1_hiera_l.yaml',
            checkpoint_path,
            device='cpu',
        )
        image_enc = sam2.image_encoder
        self.trunk = image_enc.trunk
        self.neck_convs = image_enc.neck.convs  # ModuleList len=4, reversed stage order

        self.stage_ends = self.trunk.stage_ends  # [1, 7, 43, 47]
        self.unfreeze_stages = unfreeze_stages

        # Freeze everything first
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        for p in self.neck_convs.parameters():
            p.requires_grad_(False)

        if unfreeze_stages > 0:
            n_stages = len(self.stage_ends)
            unfreeze_from_stage = n_stages - unfreeze_stages
            start_block = (
                self.stage_ends[unfreeze_from_stage - 1] + 1
                if unfreeze_from_stage > 0 else 0
            )
            for blk in self.trunk.blocks[start_block:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            if unfreeze_from_stage == 0:
                for p in self.trunk.patch_embed.parameters():
                    p.requires_grad_(True)
            # Unfreeze corresponding neck convs (convs[0] = stage3, convs[1] = stage2, ...)
            for i in range(unfreeze_stages):
                for p in self.neck_convs[i].parameters():
                    p.requires_grad_(True)

        self.strides = [4, 8, 16, 32]
        self.num_channels = [256, 256, 256, 256]

    def _freeze_until_block(self) -> int:
        n_stages = len(self.stage_ends)
        if self.unfreeze_stages <= 0:
            return len(self.trunk.blocks)
        unfreeze_from_stage = n_stages - self.unfreeze_stages
        return (
            self.stage_ends[unfreeze_from_stage - 1] + 1
            if unfreeze_from_stage > 0 else 0
        )

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        freeze_until = self._freeze_until_block()
        stage_end_set = set(self.stage_ends)
        trunk_feats: List[torch.Tensor] = []

        # Frozen prefix: patch_embed + pos_embed + early blocks
        with torch.no_grad():
            x_tok = self.trunk.patch_embed(x)
            x_tok = x_tok + self.trunk._get_pos_embed(x_tok.shape[1:3])
            for i, blk in enumerate(self.trunk.blocks[:freeze_until]):
                x_tok = blk(x_tok)
                if i in stage_end_set:
                    trunk_feats.append(x_tok.permute(0, 3, 1, 2))

        if freeze_until < len(self.trunk.blocks):
            x_tok = x_tok.detach()

        # Unfrozen suffix
        for i, blk in enumerate(self.trunk.blocks[freeze_until:], start=freeze_until):
            x_tok = blk(x_tok)
            if i in stage_end_set:
                trunk_feats.append(x_tok.permute(0, 3, 1, 2))

        # trunk_feats: [s4(144), s8(288), s16(576), s32(1152)]
        # neck_convs[3-i] maps trunk_feats[i] → 256ch
        neck_feats = [self.neck_convs[3 - i](trunk_feats[i]) for i in range(4)]

        out = {}
        for name, feat in zip(["1", "2", "3", "4"], neck_feats):
            m = tensor_list.mask
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(feat, mask)
        return out


class SAM2Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out = []
        pos = []
        for x in xs.values():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_sam2_backbone(args):
    checkpoint = getattr(args, 'sam2_checkpoint', None)
    unfreeze_stages = getattr(args, 'unfreeze_sam2_stages', 1)
    position_embedding = build_position_encoding(args)
    backbone = SAM2HieraBackbone(checkpoint, unfreeze_stages)
    return SAM2Joiner(backbone, position_embedding)
