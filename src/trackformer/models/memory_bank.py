import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """Multi-frame temporal attention for MOTR-style tracking.

    Improvements over the original:
      - mem_padding_mask: bool [B, N, K], True = empty slot, passed to cross_attn
        so padded (empty) history slots don't corrupt attention.
      - save_thresh: at inference, only update a cell's bank slot when its detection
        score exceeds this threshold (avoids polluting the bank with low-quality frames).
      - with_spatial_attn: optional spatial self-attention among all active tracks
        in the current frame (SelfMOTR MemoryBank._forward_spatial_attn).
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1, memory_len: int = 3,
                 save_thresh: float = 0.0,
                 with_spatial_attn: bool = False):
        super().__init__()
        self.memory_len = memory_len
        self.save_thresh = save_thresh

        # Temporal cross-attention: current query attends to its own K-frame history
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                 batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Optional spatial self-attention among current-frame tracks (SelfMOTR)
        self.with_spatial_attn = with_spatial_attn
        if with_spatial_attn:
            self.spatial_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                       batch_first=True)
            self.dropout_s1 = nn.Dropout(dropout)
            self.norm_s1 = nn.LayerNorm(d_model)
            self.linear_s1 = nn.Linear(d_model, dim_feedforward)
            self.dropout_s2 = nn.Dropout(dropout)
            self.linear_s2 = nn.Linear(dim_feedforward, d_model)
            self.dropout_s3 = nn.Dropout(dropout)
            self.norm_s2 = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, bank: torch.Tensor,
                mem_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tgt:              [B, N, D]    current track query embeddings
            bank:             [B, N, K, D] K past embeddings per track
            mem_padding_mask: [B, N, K]    bool, True = empty/invalid slot
        Returns:
            [B, N, D] temporally (and optionally spatially) enriched embeddings
        """
        if bank is None or bank.shape[2] == 0:
            return tgt

        B, N, K, D = bank.shape

        # Flatten batch × track dims so each track attends independently
        q  = tgt.reshape(B * N, 1, D)
        kv = bank.reshape(B * N, K, D)

        # Build key_padding_mask: [B*N, K]
        key_padding_mask = None
        if mem_padding_mask is not None:
            key_padding_mask = mem_padding_mask.reshape(B * N, K)
            # Avoid all-True rows (would produce NaN in softmax)
            all_masked = key_padding_mask.all(dim=1, keepdim=True)
            key_padding_mask = key_padding_mask & ~all_masked

        tgt2, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        tgt2 = tgt2.reshape(B, N, D)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)

        # Spatial self-attention: each track in the current frame attends to all others
        if self.with_spatial_attn and N > 1:
            tgt2, _ = self.spatial_attn(tgt, tgt, tgt)
            tgt = tgt + self.dropout_s1(tgt2)
            tgt = self.norm_s1(tgt)
            tgt2 = self.linear_s2(self.dropout_s2(F.relu(self.linear_s1(tgt))))
            tgt = tgt + self.dropout_s3(tgt2)
            tgt = self.norm_s2(tgt)

        return tgt
