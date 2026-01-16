import os
import sys
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tokenizers import Tokenizer
from datasets import load_dataset

# --- Path Fix ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.encoder import Encoder

# ================== CONFIG ==================

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_DDP = WORLD_SIZE > 1
IS_MAIN = RANK == 0
DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"

MAX_LEN = 128
EMBED_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
FF_DIM = 2048
DROPOUT = 0.1

BATCH_SIZE = 4096
EPOCHS = 5
LR = 5e-4
WARMUP_RATIO = 0.05
TEMPERATURE = 0.07

# --- PATHS ---
DATA_FOLDER = "data/processed"
TOKENIZER_PATH = "data/tokenizer/tokenizer.json"
CKPT_PATH = "train/klein_embedding_50m.pt"
LOG_FILE = "train/loss_log.txt"

# ================== TOKENIZER & COLLATOR ==================

try:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    VOCAB_SIZE = tokenizer.get_vocab_size()
except Exception as e:
    if IS_MAIN: print(f"❌ Tokenizer error: {e}")
    sys.exit(1)

def encode_batch(texts):
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
        mask = (input_ids != 0).long()
        x = self.encoder(input_ids, mask)
        mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return F.normalize(sum_embeddings / sum_mask, dim=1)

def contrastive_loss(z1, z2):
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
    if IS_DDP:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(LOCAL_RANK)
    os.makedirs("train", exist_ok=True)

    # --- LOAD LOCAL FILES ---
    if IS_MAIN: print(f"🚀 Searching for .jsonl files in {DATA_FOLDER}...")
    
    data_files = glob.glob(os.path.join(DATA_FOLDER, "*.jsonl"))
    if not data_files:
        print(f"❌ No .jsonl files found in {DATA_FOLDER}! Did you unzip the data?")
        sys.exit(1)
        
    if IS_MAIN: print(f"📂 Found {len(data_files)} files. Loading...")

    # Load dataset
    dataset = load_dataset("json", data_files=data_files, split="train")

    if IS_MAIN: print(f"✅ Loaded {len(dataset):,} pairs.")

    sampler = DistributedSampler(dataset, shuffle=True) if IS_DDP else None
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        shuffle=(sampler is None),
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True, 
        prefetch_factor=2,
        persistent_workers=True
    )

    model = SentenceEncoder().to(DEVICE)
    if IS_DDP: model = DDP(model, device_ids=[LOCAL_RANK])
    
    if IS_MAIN: print("⚡ Compiling model...")
    try:
        model = torch.compile(model)
    except:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Use standard cuda amp scaler (safest for all pytorch versions)
    scaler = torch.cuda.amp.GradScaler() 

    # --- OPEN LOG FILE (NO WANDB) ---
    log_f = None
    if IS_MAIN:
        print(f"📝 Logging metrics to {LOG_FILE}")
        log_f = open(LOG_FILE, "w", buffering=1)
        log_f.write("Step,Epoch,Loss,LR\n")

    total_steps = len(loader) * EPOCHS
    step = 0

    if IS_MAIN: print("🔥 Starting Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        if IS_DDP: sampler.set_epoch(epoch)
        for i, (xa, xb) in enumerate(loader):
            xa = xa.to(DEVICE, non_blocking=True)
            xb = xb.to(DEVICE, non_blocking=True)

            curr_lr = LR * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            for pg in optimizer.param_groups: pg['lr'] = curr_lr

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision context
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                za = model(xa)
                zb = model(xb)
                loss = contrastive_loss(za, zb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if IS_MAIN and step % 10 == 0:
                # Log to text file
                log_f.write(f"{step},{epoch},{loss.item():.6f},{curr_lr:.8f}\n")
                
                if step % 50 == 0:
                    print(f"Ep {epoch} | Step {step} | Loss {loss.item():.4f}")
            
            step += 1

    if IS_MAIN:
        save_model = model.module if IS_DDP else model
        torch.save(save_model.state_dict(), CKPT_PATH)
        print(f"✅ Saved to {CKPT_PATH}")
        if log_f: log_f.close()

    if IS_DDP: dist.destroy_process_group()

if __name__ == "__main__":
    train()