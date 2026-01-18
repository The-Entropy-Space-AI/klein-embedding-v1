import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def apply_rotary_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class SwiGLU(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class RobertaAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(q, T)
        q, k = apply_rotary_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.attn_dropout.p if self.training else 0.0)
        return self.resid_dropout(self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C)))

class RobertaBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = RobertaAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.ffn = SwiGLU(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
    def forward(self, x, attention_mask=None):
        x = self.norm1(x + self.attn(x, attention_mask))
        return self.norm2(x + self.ffn(x))

class RobertaSentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(2, embed_dim)
        self.layers = nn.ModuleList([RobertaBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.LayerNorm = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(self, input_ids, attention_mask=None):
        x = self.word_embeddings(input_ids) + self.token_type_embeddings(torch.zeros_like(input_ids))
        x = self.LayerNorm(x)
        if attention_mask is not None:
            mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
        else: mask = None
        for layer in self.layers: x = layer(x, mask)
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
            x = torch.sum(x * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        else: x = x.mean(dim=1)
        return F.normalize(x, dim=1)