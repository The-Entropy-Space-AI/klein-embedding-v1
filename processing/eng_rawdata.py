from datasets import load_dataset
from tqdm import tqdm

print("Downloading English corpus...")
ds = load_dataset("cc_news", split="train", streaming=True)

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