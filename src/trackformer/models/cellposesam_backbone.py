"""CellposeSAM ViT-L image encoder as a multi-scale Deformable DETR backbone.

Self-contained implementation — no segment_anything package required.
SAM ViT-L building blocks are reproduced here verbatim from the original SAM
source (Apache-2.0 / Meta) so this file has zero extra dependencies.

Architecture overview
---------------------
  CellposeSAM ViT-L encoder (24 blocks, embed_dim=1024, stride-8 patches)
       ↓  neck → [B, 256, H/8, W/8]           (level 0, stride 8)
       ↓  down1 (stride-2 conv) → [B, 256, H/16, W/16]  (level 1, stride 16)
       ↓  down2 (stride-2 conv) → [B, 256, H/32, W/32]  (level 2, stride 32)

The encoder is frozen by default; only down1/down2 are trained.

Config keys (add to YAML):
  backbone: cellposesam
  cellposesam_checkpoint: /srv/home/chen/.cellpose/models/cpsam   # default
  freeze_cellposesam_encoder: true                                  # default
"""

from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..util.misc import NestedTensor
from .position_encoding import build_position_encoding
from .backbone import Joiner

_DEFAULT_CPSAM_CHECKPOINT = "/srv/home/chen/.cellpose/models/cpsam"

# ---------------------------------------------------------------------------
# SAM ViT-L building blocks (reproduced from Meta's segment-anything repo,
# Apache-2.0 license, https://github.com/facebookresearch/segment-anything)
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int,
                 act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size: Tuple[int, int] = (16, 16),
                 stride: Tuple[int, int] = (16, 16),
                 padding: Tuple[int, int] = (0, 0),
                 in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)  # B C H W → B H W C


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor, q: torch.Tensor,
    rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int], k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    return (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 use_rel_pos: bool = False, rel_pos_zero_init: bool = True,
                 input_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, dim // num_heads))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, dim // num_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition [B,H,W,C] into non-overlapping [B*nW, ws, ws, C] windows."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C), (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """Reconstruct [B,H,W,C] from windowed [B*nW, ws, ws, C]."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    return x[:, :H, :W, :].contiguous()


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, norm_layer: Type[nn.Module] = nn.LayerNorm,
                 act_layer: Type[nn.Module] = nn.GELU, use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True, window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
                              input_size=input_size if window_size == 0 else (window_size, window_size))
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoderViT(nn.Module):
    """SAM image encoder, instantiated for ViT-L with CellposeSAM's stride-8 patches.

    Rel-pos parameter sizes follow the original SAM ViT-L training convention:
      - global attention blocks (5,11,17,23): input_size=(64,64) → rel_pos [127, head_dim]
      - windowed blocks (all others):         input_size=(14,14) → rel_pos [27,  head_dim]
    All blocks use window_size=0 (full global attention), matching CellposeSAM's fine-tuning.
    If GPU memory is limited, switch non-global blocks to window_size=14.
    get_rel_pos() interpolates to the actual token grid at forward time.
    """

    # SAM ViT-L architecture constants
    _GLOBAL_ATTN_INDEXES = (5, 11, 17, 23)
    _WINDOW_SIZE = 14     # windowed blocks input_size → rel_pos [27, head_dim]
    _GLOBAL_GRID = 64     # global blocks input_size  → rel_pos [127, head_dim]

    def __init__(self, patch_size: int = 8, embed_dim: int = 1024, depth: int = 24,
                 num_heads: int = 16, mlp_ratio: float = 4.0, out_chans: int = 256,
                 qkv_bias: bool = True, trained_grid_size: int = 32) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=3, embed_dim=embed_dim,
        )
        # Absolute position embedding sized for the CellposeSAM training grid
        # (32×32 for 256px images at stride 8)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, trained_grid_size, trained_grid_size, embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                  use_rel_pos=True, rel_pos_zero_init=True,
                  window_size=0,  # all global, matching CellposeSAM fine-tuning
                  # input_size determines rel_pos parameter shape to match checkpoint
                  input_size=(
                      (self._GLOBAL_GRID, self._GLOBAL_GRID)
                      if i in self._GLOBAL_ATTN_INDEXES
                      else (self._WINDOW_SIZE, self._WINDOW_SIZE)
                  ))
            for i in range(depth)
        ])

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # [B, H/ps, W/ps, embed_dim]

        # Interpolate stored pos_embed to the actual token grid size
        h_tok, w_tok = x.shape[1], x.shape[2]
        pe = self.pos_embed  # [1, G, G, embed_dim]
        if pe.shape[1] != h_tok or pe.shape[2] != w_tok:
            pe = pe.permute(0, 3, 1, 2)  # [1, embed_dim, G, G]
            pe = F.interpolate(pe, size=(h_tok, w_tok), mode='bilinear', align_corners=False)
            pe = pe.permute(0, 2, 3, 1)  # [1, h_tok, w_tok, embed_dim]
        x = x + pe

        for blk in self.blocks:
            x = blk(x)

        return self.neck(x.permute(0, 3, 1, 2))  # [B, out_chans, H/ps, W/ps]


# ---------------------------------------------------------------------------
# CellposeSAM backbone wrapper
# ---------------------------------------------------------------------------

class CellposeSAMBackbone(nn.Module):
    """Wraps CellposeSAM's ViT-L encoder as a 4-level Deformable DETR backbone.

    Outputs feature maps at strides 4, 8, 16, 32 — all with 256 channels.
    The ViT-L encoder is partially frozen: early blocks (0..depth-unfreeze_blocks-1)
    are frozen; the last `unfreeze_blocks` blocks and the neck are trainable.
    A ViTDet-style FPN creates genuine multi-scale features from the stride-8 neck
    output: one upsampled stride-4 level (fine spatial detail) and two downsampled
    levels (stride-16, stride-32).
    """

    def __init__(self, checkpoint_path: str = _DEFAULT_CPSAM_CHECKPOINT,
                 freeze_encoder: bool = True,
                 unfreeze_blocks: int = 8) -> None:
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder = ImageEncoderViT()

        # Load pretrained CellposeSAM weights into encoder
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        enc_state = {k[len("encoder."):]: v
                     for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=True)
        if missing:
            raise RuntimeError(f"CellposeSAM encoder: missing keys {missing[:5]}")

        n_blocks = len(self.encoder.blocks)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            # Unfreeze the last unfreeze_blocks ViT blocks and the neck so they
            # can adapt their features toward detection.
            self.unfreeze_from = n_blocks - unfreeze_blocks
            for blk in self.encoder.blocks[self.unfreeze_from:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for p in self.encoder.neck.parameters():
                p.requires_grad_(True)
            # Switch all unfrozen blocks to windowed attention (window_size=14).
            # Global attention on 5475 tokens (584×600 at stride-8) needs 1.79 GiB per
            # block; storing that for backward would OOM. Windowed attention (196 tokens
            # per window) uses ~88 MB per block instead. get_rel_pos() interpolates the
            # stored rel_pos to the actual window grid, so the checkpoint is compatible.
            for blk in self.encoder.blocks[self.unfreeze_from:]:
                blk.window_size = 14
        else:
            self.unfreeze_from = 0  # all blocks trainable

        # ViTDet-style FPN: create 4 levels from the single stride-8 neck output.
        # up1:   stride-8 → stride-4  (deconv upsample + spatial mixing conv)
        # down1: stride-8 → stride-16 (strided conv)
        # down2: stride-16 → stride-32 (strided conv)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )

        # len=5 → num_backbone_outs = len-1 = 4, matching num_feature_levels=4
        self.strides = [2, 4, 8, 16, 32]
        self.num_channels = [256, 256, 256, 256, 256]

    def forward(self, tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
        x = tensor_list.tensors  # [B, 3, H, W]

        if self.freeze_encoder:
            # Frozen blocks: run under no_grad to avoid storing activations
            with torch.no_grad():
                x = self.encoder.patch_embed(x)
                h_tok, w_tok = x.shape[1], x.shape[2]
                pe = self.encoder.pos_embed
                if pe.shape[1] != h_tok or pe.shape[2] != w_tok:
                    pe = pe.permute(0, 3, 1, 2)
                    pe = F.interpolate(pe, size=(h_tok, w_tok), mode='bilinear', align_corners=False)
                    pe = pe.permute(0, 2, 3, 1)
                x = x + pe
                for blk in self.encoder.blocks[:self.unfreeze_from]:
                    x = blk(x)
            # Detach: stop gradients from flowing into the frozen region
            x = x.detach()
            # Trainable tail blocks: gradient checkpointing avoids storing all
            # intermediate activations simultaneously (recomputes during backward).
            for blk in self.encoder.blocks[self.unfreeze_from:]:
                x = checkpoint(blk, x, use_reentrant=False)
            feat0 = self.encoder.neck(x.permute(0, 3, 1, 2))  # [B, 256, H/8, W/8]
        else:
            feat0 = self.encoder(x)  # [B, 256, H/8, W/8]

        feat_s4  = self.up1(feat0)          # [B, 256, H/4,  W/4]
        feat_s16 = self.down1(feat0)        # [B, 256, H/16, W/16]
        feat_s32 = self.down2(feat_s16)     # [B, 256, H/32, W/32]

        out: Dict[str, NestedTensor] = {}
        for name, feat in [("1", feat_s4), ("2", feat0), ("3", feat_s16), ("4", feat_s32)]:
            m = tensor_list.mask
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(feat, mask)
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_cellposesam_backbone(args) -> Joiner:
    position_embedding = build_position_encoding(args)
    backbone = CellposeSAMBackbone(
        checkpoint_path=getattr(args, 'cellposesam_checkpoint', _DEFAULT_CPSAM_CHECKPOINT),
        freeze_encoder=getattr(args, 'freeze_cellposesam_encoder', True),
        unfreeze_blocks=getattr(args, 'unfreeze_cellposesam_blocks', 8),
    )
    model = Joiner(backbone, position_embedding)
    model.strides = backbone.strides
    model.num_channels = backbone.num_channels
    return model
