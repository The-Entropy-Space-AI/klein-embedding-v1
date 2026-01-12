from tokenizers import ByteLevelBPETokenizer
import os

files = [
    "data/raw/english.txt",
    "data/raw/hindi_native.txt",
    "data/raw/hindi_roman.txt",
    "data/raw/malayalam_native.txt",
    "data/raw/malayalam_roman.txt",
    "data/raw/tamil_native.txt",
    "data/raw/tamil_roman.txt"
]

def batch_iterator(file_paths, limit=50000):
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                yield line

tokenizer = ByteLevelBPETokenizer()

tokenizer.train_from_iterator(
    batch_iterator(files),
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

os.makedirs("data/tokenizer", exist_ok=True)
tokenizer.save_model("data/tokenizer")