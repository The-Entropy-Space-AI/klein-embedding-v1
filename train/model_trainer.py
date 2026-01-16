import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tokenizers import Tokenizer
import wandb

# --- FIX: Add project root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.encoder import Encoder

# ================== CONFIG ==================
# Detect Distributed Environment
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_DDP = WORLD_SIZE > 1
IS_MAIN = RANK == 0
DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"

# Hyperparameters
MAX_LEN = 128
EMBED_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
FF_DIM = 2048
DROPOUT = 0.1

# Batch size per GPU (Effective batch size = BATCH_SIZE * WORLD_SIZE)
# 4096 fits on L40S (48GB) or H100 (80GB). 
BATCH_SIZE = 4096 
EPOCHS = 5
LR = 5e-4
WARMUP_RATIO = 0.05
TEMPERATURE = 0.07

DATA_PATH = "data/processed/samanantar_train.jsonl"
TOKENIZER_PATH = "data/tokenizer/tokenizer.json"
CKPT_PATH = "train/klein_embedding_50m.pt"
LOSS_LOG = "train/loss.txt"

# ================== SETUP ==================

def setup():
    if IS_DDP:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(LOCAL_RANK)

def cleanup():
    if IS_DDP:
        dist.destroy_process_group()

# ================== COMPONENTS ==================

class PairDataset(Dataset):
    def __init__(self, path):
        self.pairs = []
        if IS_MAIN: print(f"Loading data from {path}...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    self.pairs.append((obj["text_a"], obj["text_b"]))
                except json.JSONDecodeError: continue

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

try:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    VOCAB_SIZE = tokenizer.get_vocab_size()
except Exception as e:
    if IS_MAIN: print(f"Error loading tokenizer: {e}")
    sys.exit(1)

def encode(text: str):
    if text is None: text = ""
    ids = tokenizer.encode(text).ids[:MAX_LEN]
    if len(ids) < MAX_LEN: ids += [0] * (MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)

class SentenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=VOCAB_SIZE, max_len=MAX_LEN, embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS, ff_dim=FF_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT
        )

    def forward(self, input_ids):
        mask = (input_ids != 0).long() # Create padding mask
        x = self.encoder(input_ids, mask)
        
        # Mean Pooling with Mask
        mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return F.normalize(sum_embeddings / sum_mask, dim=1)

def contrastive_loss(z1, z2):
    # Gather all embeddings if using multiple GPUs to calculate accurate global loss
    if IS_DDP:
        z1_list = [torch.zeros_like(z1) for _ in range(WORLD_SIZE)]
        z2_list = [torch.zeros_like(z2) for _ in range(WORLD_SIZE)]
        dist.all_gather(z1_list, z1)
        dist.all_gather(z2_list, z2)
        z1_all = torch.cat(z1_list)
        z2_all = torch.cat(z2_list)
    else:
        z1_all, z2_all = z1, z2
        
    logits = z1_all @ z2_all.T / TEMPERATURE
    labels = torch.arange(z1_all.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)

# ================== TRAINING LOOP ==================

def train():
    setup()
    
    # Dataset & Sampler
    dataset = PairDataset(DATA_PATH)
    sampler = DistributedSampler(dataset, shuffle=True) if IS_DDP else None
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        shuffle=(sampler is None),
        num_workers=8,        # High workers for fast GPU
        pin_memory=True, 
        prefetch_factor=2
    )

    # Model
    model = SentenceEncoder().to(DEVICE)
    if IS_DDP:
        model = DDP(model, device_ids=[LOCAL_RANK])
    
    # H100/L40S Optimization: Compile
    if IS_MAIN: print("Compiling model...")
    try:
        model = torch.compile(model)
    except:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision

    # Logging
    if IS_MAIN:
        wandb.init(project="klein-embedding-50m", config={"batch_size": BATCH_SIZE, "lr": LR, "world_size": WORLD_SIZE})
        print(f"Training on {WORLD_SIZE} GPU(s). Total Batch Size: {BATCH_SIZE * WORLD_SIZE}")

    total_steps = len(loader) * EPOCHS
    step = 0

    model.train()
    for epoch in range(EPOCHS):
        if IS_DDP: sampler.set_epoch(epoch)
        
        for i, (text_a, text_b) in enumerate(loader):
            # Move to device non-blocking
            xa = torch.stack([encode(t) for t in text_a]).to(DEVICE, non_blocking=True)
            xb = torch.stack([encode(t) for t in text_b]).to(DEVICE, non_blocking=True)

            # Cosine LR
            curr_lr = LR * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            for param_group in optimizer.param_groups: param_group['lr'] = curr_lr

            optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Forward
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                za = model(xa)
                zb = model(xb)
                loss = contrastive_loss(za, zb)

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if IS_MAIN and step % 10 == 0:
                wandb.log({"loss": loss.item(), "lr": curr_lr, "epoch": epoch}, step=step)
                if step % 50 == 0:
                    print(f"Ep {epoch} | Step {step} | Loss {loss.item():.4f}")

            step += 1

    if IS_MAIN:
        save_model = model.module if IS_DDP else model
        torch.save(save_model.state_dict(), CKPT_PATH)
        print("✓ Training Complete. Model Saved.")
        wandb.finish()

    cleanup()

if __name__ == "__main__":
    train()