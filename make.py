import json

# The exact code blocks you provided, organized by cell
cells = [
    # Cell 1: Imports
    r"""# ============================================================
# CELL 1: Imports and Setup
# ============================================================

import os
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tokenizers import Tokenizer
import wandb
""",
    # Cell 2: DDP Setup
    r"""# ============================================================
# CELL 2: DDP Setup Functions
# ============================================================

def setup_ddp():
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_rank(), dist.get_world_size()
    else:
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
""",
    # Cell 3: Attention
    r"""# ============================================================
# CELL 3: RoBERTa-style Multi-Head Attention
# ============================================================

class RobertaAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out
""",
    # Cell 4: Block
    r"""# ============================================================
# CELL 4: RoBERTa Encoder Block (Post-norm)
# ============================================================

class RobertaBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = RobertaAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        
        # Intermediate layer (GELU activation)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(self, x, attention_mask=None):
        # Post-norm: residual -> attention -> norm
        attn_out = self.attn(x, attention_mask)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
""",
    # Cell 5: Embeddings
    r"""# ============================================================
# CELL 5: RoBERTa Embeddings
# ============================================================

class RobertaEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, pad_token_id=0, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_len + 2, embed_dim)  # +2 for padding offset
        self.token_type_embeddings = nn.Embedding(2, embed_dim)  # For NSP-style tasks
        
        self.LayerNorm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        
        self.padding_idx = pad_token_id
        self.position_embedding_type = "absolute"

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        B, T = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)
            position_ids = position_ids + 2  # RoBERTa offset
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
""",
    # Cell 6: Encoder
    r"""# ============================================================
# CELL 6: RoBERTa Encoder
# ============================================================

class RobertaEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        
        self.embeddings = RobertaEmbeddings(vocab_size, max_len, embed_dim, dropout=dropout)
        
        self.layers = nn.ModuleList([
            RobertaBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids, attention_mask=None):
        # Create attention mask in correct format
        if attention_mask is not None:
            # Convert to (B, 1, 1, T) and invert (1 -> 0, 0 -> -inf)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        x = self.embeddings(input_ids)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        return x
""",
    # Cell 7: Sentence Encoder
    r"""# ============================================================
# CELL 7: RoBERTa Sentence Encoder
# ============================================================

class RobertaSentenceEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.encoder = RobertaEncoder(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.encoder(input_ids, attention_mask)
        
        # Mean pooling (excluding padding)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
            sum_embeddings = torch.sum(x * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            x = sum_embeddings / sum_mask
        else:
            x = x.mean(dim=1)
        
        return F.normalize(x, dim=1)
""",
    # Cell 8: Dataset
    r"""# ============================================================
# CELL 8: Dataset
# ============================================================

class PairDataset(Dataset):
    def __init__(self, path):
        self.pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    self.pairs.append((obj["text_a"], obj["text_b"]))
                except json.JSONDecodeError:
                    continue

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
""",
    # Cell 9: Loss
    r"""# ============================================================
# CELL 9: Loss and Learning Rate Scheduler
# ============================================================

def contrastive_loss(z1, z2, temperature):
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def cosine_lr(step, total_steps, lr, warmup_ratio):
    warmup_steps = int(warmup_ratio * total_steps)
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr * 0.5 * (1 + math.cos(math.pi * progress))
""",
    # Cell 10: Config
    r"""# ============================================================
# CELL 10: Configuration (50M params RoBERTa-base scale)
# ============================================================

# Paths
TOKENIZER_PATH = "/kaggle/input/your-data/tokenizer.json"
DATA_PATH = "/kaggle/input/your-data/samanantar_train.jsonl"

# RoBERTa-base config (scaled to 50M)
MAX_LEN = 128
EMBED_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
FF_DIM = 2048  # 4x embed_dim (RoBERTa standard)
DROPOUT = 0.1

# Training config
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4  # RoBERTa uses lower LR
WARMUP_RATIO = 0.06  # RoBERTa-style warmup
TEMPERATURE = 0.07
""",
    # Cell 11: DDP Init
    r"""# ============================================================
# CELL 11: Initialize DDP and Load Tokenizer
# ============================================================

local_rank, rank, world_size = setup_ddp()
DEVICE = f'cuda:{local_rank}'
is_main = rank == 0

if is_main:
    print(f"Using {world_size} GPU(s)")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.get_vocab_size()

if is_main:
    print(f"Tokenizer vocab size: {VOCAB_SIZE}")

def encode(text: str):
    if text is None:
        text = ""
    ids = tokenizer.encode(text).ids[:MAX_LEN]
    mask = [1] * len(ids)
    
    # Pad to MAX_LEN
    if len(ids) < MAX_LEN:
        padding_length = MAX_LEN - len(ids)
        ids += [0] * padding_length
        mask += [0] * padding_length
    
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
""",
    # Cell 12: Model Init
    r"""# ============================================================
# CELL 12: Initialize Model with DDP
# ============================================================

model = RobertaSentenceEncoder(
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

if is_main:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
""",
    # Cell 13: Data Loaders
    r"""# ============================================================
# CELL 13: Setup Data Loaders
# ============================================================

dataset = PairDataset(DATA_PATH)
if is_main:
    print(f"Loaded {len(dataset)} pairs")

sampler = DistributedSampler(
    dataset, 
    num_replicas=world_size, 
    rank=rank, 
    shuffle=True
) if world_size > 1 else None

loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,
    shuffle=(sampler is None),
    num_workers=2,
    pin_memory=True
)
""",
    # Cell 14: Optimizer
    r"""# ============================================================
# CELL 14: Setup Optimizer and W&B
# ============================================================

# RoBERTa-style optimizer (AdamW with weight decay)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LR,
    betas=(0.9, 0.98),
    eps=1e-6,
    weight_decay=0.01
)

total_steps = EPOCHS * len(loader)

if is_main:
    wandb.login()
    run = wandb.init(
        project="klein-roberta-50m",
        config={
            "architecture": "RoBERTa",
            "vocab_size": VOCAB_SIZE,
            "max_len": MAX_LEN,
            "embed_dim": EMBED_DIM,
            "layers": NUM_LAYERS,
            "heads": NUM_HEADS,
            "ff_dim": FF_DIM,
            "dropout": DROPOUT,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "temperature": TEMPERATURE,
            "world_size": world_size,
            "total_params": total_params,
        },
    )
""",
    # Cell 15: Training Loop
    r"""# ============================================================
# CELL 15: Training Loop
# ============================================================

model.train()
step = 0

for epoch in range(EPOCHS):
    if sampler:
        sampler.set_epoch(epoch)
    
    epoch_loss = 0.0
    
    for i, (text_a, text_b) in enumerate(loader):
        # Encode with attention masks
        ids_a, mask_a = zip(*[encode(t) for t in text_a])
        ids_b, mask_b = zip(*[encode(t) for t in text_b])
        
        xa = torch.stack(ids_a).to(DEVICE)
        xb = torch.stack(ids_b).to(DEVICE)
        mask_a = torch.stack(mask_a).to(DEVICE)
        mask_b = torch.stack(mask_b).to(DEVICE)

        za = model(xa, mask_a)
        zb = model(xb, mask_b)

        loss = contrastive_loss(za, zb, TEMPERATURE)

        lr = cosine_lr(step, total_steps, LR, WARMUP_RATIO)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()

        epoch_loss += loss.item()

        if is_main:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/epoch": epoch,
                },
                step=step,
            )

            if step % 50 == 0:
                print(f"Epoch {epoch} | step {step} | loss {loss.item():.4f} | lr {lr:.2e}")

        step += 1
    
    if is_main:
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")
""",
    # Cell 16: Save
    r"""# ============================================================
# CELL 16: Save Model and Cleanup
# ============================================================

if is_main:
    save_model = model.module if isinstance(model, DDP) else model
    torch.save(save_model.state_dict(), "roberta_encoder_50m.pt")
    print("✓ Model saved")

    artifact = wandb.Artifact("roberta-model-50m", type="model")
    artifact.add_file("roberta_encoder_50m.pt")
    run.log_artifact(artifact)
    wandb.finish()

cleanup_ddp()
"""
]

# Construct the notebook JSON structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Add cells to the notebook
for source_code in cells:
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_code.splitlines(True)  # Split into lines but keep newlines
    }
    notebook["cells"].append(cell)

# Write to file
filename = "roberta_training.ipynb"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Successfully created {filename}")