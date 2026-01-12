from datasets import load_dataset
import os
from tqdm import tqdm

os.makedirs("data/raw", exist_ok=True)

langs = [
    ("tam_Taml", "tamil_native.txt"),
    ("tam_Latn", "tamil_roman.txt"),
    ("hin_Deva", "hindi_native.txt"),
    ("hin_Latn", "hindi_roman.txt"),
    ("mal_Mlym", "malayalam_native.txt"),
    ("mal_Latn", "malayalam_roman.txt"),
]

MAX_SAMPLES = 500_000

print("Downloading Indic languages...")
for lang_code, filename in langs:
    print(f"Processing {lang_code}...")
    ds = load_dataset(
        "ai4bharat/sangraha", 
        data_dir=f"synthetic/{lang_code}",
        split="train",
        streaming=True
    )
    
    with open(f"data/raw/{filename}", "w", encoding="utf-8", errors="ignore") as f:
        for i, row in enumerate(tqdm(ds, desc=filename, total=MAX_SAMPLES)):
            if i >= MAX_SAMPLES:
                break
            text = row["text"].strip()
            if text:
                try:
                    f.write(text + "\n")
                except:
                    continue

print("\nDownloading English Wikipedia...")
ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

with open("data/raw/english.txt", "w", encoding="utf-8", errors="ignore") as f:
    for i, row in enumerate(tqdm(ds, desc="english.txt", total=3_000_000)):
        if i >= 3_000_000:
            break
        text = row["text"].strip()
        if text:
            for line in text.split("\n"):
                line = line.strip()
                if line:
                    try:
                        f.write(line + "\n")
                    except:
                        continue

print("Done!")