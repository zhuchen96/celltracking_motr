import torch
import torch.nn as nn


class MemoryBank(nn.Module):
    """Multi-frame temporal attention for MOTR-style tracking.

    Each of the N track queries cross-attends to its own K past decoder output
    embeddings, aggregating temporal context before it enters the main decoder.
    Applied *before* QIM so the history-enriched embedding is then further
    contextualized by the current-frame encoder memory.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1, memory_len: int = 3):
        super().__init__()
        self.memory_len = memory_len

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

    def forward(self, tgt: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt:  [B, N, D]    current track query embeddings
            bank: [B, N, K, D] K past embeddings per track
        Returns: [B, N, D] temporally-enriched embeddings
        """
        if bank is None or bank.shape[2] == 0:
            return tgt

        B, N, K, D = bank.shape
        # Each track query attends independently to its own K memories
        q = tgt.reshape(B * N, 1, D)
        kv = bank.reshape(B * N, K, D)

        tgt2, _ = self.cross_attn(q, kv, kv)
        tgt2 = tgt2.reshape(B, N, D)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)
        return tgt
