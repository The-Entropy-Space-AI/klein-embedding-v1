import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tokenizers import Tokenizer
import wandb
from datasets import load_dataset

# --- Path Fix for project imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.encoder import Encoder

# ================== CONFIG ==================

# Environment
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_DDP = WORLD_SIZE > 1
IS_MAIN = RANK == 0
DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"

# Model Config (50M Scale)
MAX_LEN = 128
EMBED_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
FF_DIM = 2048
DROPOUT = 0.1

# Training Config
# 4096 fits easily in L40S (48GB VRAM). 
BATCH_SIZE = 4096       
EPOCHS = 5
LR = 5e-4
WARMUP_RATIO = 0.05
TEMPERATURE = 0.07

# --- UPDATED PATHS ---
# Your HF Dataset Repo ID
HF_DATASET_ID = "the-entropy-space-ai/klein-embedding-data"
TOKENIZER_PATH = "data/tokenizer/tokenizer.json"
CKPT_PATH = "train/klein_embedding_50m.pt"

# ================== TOKENIZER & COLLATOR ==================

try:
    # Ensure you have your tokenizer file locally (git pull if needed)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    VOCAB_SIZE = tokenizer.get_vocab_size()
except Exception as e:
    if IS_MAIN: print(f"❌ Error loading tokenizer from {TOKENIZER_PATH}: {e}")
    sys.exit(1)

def encode_batch(texts):
    """Batched tokenization helper for speed."""
    if not texts: return torch.zeros((0, MAX_LEN), dtype=torch.long)
    
    encodings = tokenizer.encode_batch(texts)
    
    batch_ids = []
    for enc in encodings:
        ids = enc.ids[:MAX_LEN]
        if len(ids) < MAX_LEN:
            ids += [0] * (MAX_LEN - len(ids))
        batch_ids.append(ids)
    
    return torch.tensor(batch_ids, dtype=torch.long)

def collate_fn(batch):
    """
    Runs on CPU workers. Prepares tensors so GPU doesn't wait.
    """
    text_a = [item['text_a'] or "" for item in batch]
    text_b = [item['text_b'] or "" for item in batch]
    
    xa = encode_batch(text_a)
    xb = encode_batch(text_b)
    
    return xa, xb

# ================== MODEL ==================

class SentenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=VOCAB_SIZE, max_len=MAX_LEN, embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS, ff_dim=FF_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT
        )

    def forward(self, input_ids):
        # Create mask (1 for tokens, 0 for padding)
        mask = (input_ids != 0).long()
        x = self.encoder(input_ids, mask)
        
        # Mean Pooling with Mask
        mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return F.normalize(sum_embeddings / sum_mask, dim=1)

def contrastive_loss(z1, z2):
    if IS_DDP:
        # Use simple DDP loss (local batch positives vs local+remote negatives)
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
    if IS_DDP:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(LOCAL_RANK)
        
    os.makedirs("train", exist_ok=True)

    # --- LOAD FROM HUGGING FACE ---
    if IS_MAIN: print(f"🚀 Loading dataset from HF: {HF_DATASET_ID}...")
    
    # This pulls from your uploaded repo. 
    try:
        # Tries to load as a standard dataset
        dataset = load_dataset(HF_DATASET_ID, split="train")
    except Exception as e:
        if IS_MAIN: print(f"⚠️ Standard load failed ({e}), trying generic JSON load...")
        # Fallback: construct the direct URL to the file
        data_url = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main/samanantar_train.jsonl"
        dataset = load_dataset("json", data_files=data_url, split="train")

    if IS_MAIN: print(f"✅ Loaded {len(dataset):,} pairs.")

    sampler = DistributedSampler(dataset, shuffle=True) if IS_DDP else None
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        shuffle=(sampler is None),
        num_workers=16,           # L40S has plenty of cores
        collate_fn=collate_fn,    # Tokenize in parallel
        pin_memory=True, 
        prefetch_factor=2,
        persistent_workers=True
    )

    # --- MODEL SETUP ---
    model = SentenceEncoder().to(DEVICE)
    if IS_DDP: model = DDP(model, device_ids=[LOCAL_RANK])
    
    if IS_MAIN: print("⚡ Compiling model for speed...")
    try:
        model = torch.compile(model)
    except:
        pass # Fallback for older PyTorch

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    if IS_MAIN:
        wandb.init(project="klein-embedding-50m", config={"batch_size": BATCH_SIZE, "lr": LR, "world_size": WORLD_SIZE})

    total_steps = len(loader) * EPOCHS
    step = 0

    if IS_MAIN: print("🔥 Starting Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        if IS_DDP: sampler.set_epoch(epoch)
        
        # Loop now receives PRE-TOKENIZED tensors (thanks to collate_fn)
        for i, (xa, xb) in enumerate(loader):
            xa = xa.to(DEVICE, non_blocking=True)
            xb = xb.to(DEVICE, non_blocking=True)

            # Cosine LR Schedule
            curr_lr = LR * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            for pg in optimizer.param_groups: pg['lr'] = curr_lr

            optimizer.zero_grad(set_to_none=True)

            # Mixed Precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                za = model(xa)
                zb = model(xb)
                loss = contrastive_loss(za, zb)

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
        print(f"✅ Saved 50M Model to {CKPT_PATH}")
        wandb.finish()

    if IS_DDP: dist.destroy_process_group()

if __name__ == "__main__":
    train()