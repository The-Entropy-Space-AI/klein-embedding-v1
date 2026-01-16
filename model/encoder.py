import torch
import torch.nn as nn
from .mha import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_len, embed_dim)
        
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, mask=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.embed_norm(x)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, mask=mask)
        
        return self.final_norm(x)