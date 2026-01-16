import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if mask is not None:
            mask = mask.view(B, 1, 1, T) 
            pass

        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None if mask is None else mask.bool(),
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)