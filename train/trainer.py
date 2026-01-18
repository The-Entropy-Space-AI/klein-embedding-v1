import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

# Internal Imports
from model.roberta import RobertaSentenceEncoder
from processing.dataset import PairDataset, get_encode_fn

# ============================================================
# DDP & Utility Functions
# ============================================================

def setup_ddp():
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_rank(), dist.get_world_size()
    return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def contrastive_loss(z1, z2, temperature=0.05):
    logits = z1 @ z2.T / temperature
    labels = torch.arange(len(z1), device=z1.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    alignment_loss = F.mse_loss(z1 @ z1.T / temperature, z2 @ z2.T / temperature)
    return loss + 0.5 * alignment_loss

def cosine_lr(step, total_steps, lr, warmup_ratio):
    warmup_steps = int(warmup_ratio * total_steps)
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr * 0.5 * (1 + math.cos(math.pi * progress))

# ============================================================
# Training Configuration
# ============================================================

TOKENIZER_PATH = "tokenizer.json" # Update this path
DATA_PATH = "data/"              # Update this path
MAX_LEN = 128
EMBED_DIM = 480
NUM_LAYERS = 7
NUM_HEADS = 8
FF_DIM = 1920
BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-4
WARMUP_RATIO = 0.06
MAX_PAIRS = 1000000

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    local_rank, rank, world_size = setup_ddp()
    DEVICE = f'cuda:{local_rank}'
    is_main = rank == 0

    encode, VOCAB_SIZE = get_encode_fn(TOKENIZER_PATH, MAX_LEN)

    model = RobertaSentenceEncoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    dataset = PairDataset(DATA_PATH, max_pairs=MAX_PAIRS, balanced=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=(sampler is None), num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    total_steps = EPOCHS * len(loader)
    
    if is_main:
        writer = SummaryWriter(log_dir=f"runs/klein_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    model.train()
    step = 0
    for epoch in range(EPOCHS):
        if sampler: sampler.set_epoch(epoch)
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", disable=not is_main)
        
        for text_a, text_b in pbar:
            ids_a, mask_a = zip(*[encode(t) for t in text_a])
            ids_b, mask_b = zip(*[encode(t) for t in text_b])
            
            xa, ma = torch.stack(ids_a).to(DEVICE), torch.stack(mask_a).to(DEVICE)
            xb, mb = torch.stack(ids_b).to(DEVICE), torch.stack(mask_b).to(DEVICE)

            with torch.amp.autocast('cuda'):
                za, zb = model(xa, ma), model(xb, mb)
                loss = contrastive_loss(za, zb)

            lr = cosine_lr(step, total_steps, LR, WARMUP_RATIO)
            for g in optimizer.param_groups: g["lr"] = lr

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if is_main:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})
                writer.add_scalar('train/loss', loss.item(), step)
            step += 1

    if is_main:
        torch.save(model.state_dict(), "roberta_klein_v1.pt")
        writer.close()
    
    cleanup_ddp()