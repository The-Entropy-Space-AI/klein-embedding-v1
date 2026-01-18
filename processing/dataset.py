import glob, json, random, torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class PairDataset(Dataset):
    def __init__(self, data_dir, max_pairs=None, balanced=True):
        self.pairs = []
        files = sorted(glob.glob(f"{data_dir}/samanantar_part_*.jsonl"))
        pairs_per_file = max_pairs // len(files) if max_pairs and balanced else None
        
        for file_path in files:
            with open(file_path, encoding="utf-8") as f:
                file_pairs = [tuple(json.loads(line).values()) for line in f if line.strip()]
            if pairs_per_file and len(file_pairs) > pairs_per_file:
                file_pairs = random.sample(file_pairs, pairs_per_file)
            self.pairs.extend(file_pairs)
            if max_pairs and not balanced and len(self.pairs) >= max_pairs: break
        print(f"Total pairs: {len(self.pairs)}")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

def get_encode_fn(tokenizer_path, max_len):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    def encode(text):
        res = tokenizer.encode(text or "")
        ids = res.ids[:max_len]
        mask = [1] * len(ids)
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
            mask += [0] * (max_len - len(ids))
        return torch.tensor(ids), torch.tensor(mask)
    return encode, tokenizer.get_vocab_size()