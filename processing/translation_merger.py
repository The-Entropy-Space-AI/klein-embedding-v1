import json
from pathlib import Path

# -------- CONFIG --------
PAIRS = [("en", "ta"), ("en", "hi"), ("en", "ml")]

RAW_ROOT = Path("data/raw/idk")
OUT_PATH = Path("data/processed/samanantar_train.jsonl")

MAX_LEN_CHARS = 500
MIN_LEN_CHARS = 3
LEN_RATIO_LIMIT = 3.0
# ------------------------


def clean(text: str) -> str:
    text = text.strip()
    text = " ".join(text.split())
    return text


def valid_pair(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if len(a) < MIN_LEN_CHARS or len(b) < MIN_LEN_CHARS:
        return False
    if len(a) > MAX_LEN_CHARS or len(b) > MAX_LEN_CHARS:
        return False

    ratio = max(len(a), len(b)) / max(1, min(len(a), len(b)))
    if ratio > LEN_RATIO_LIMIT:
        return False

    return True


def merge_pair(lang_a: str, lang_b: str, out_f):
    pair_dir = RAW_ROOT / f"{lang_a}-{lang_b}"

    file_a = pair_dir / f"train.{lang_a}"
    file_b = pair_dir / f"train.{lang_b}"

    print(f"Merging {file_a} + {file_b}")

    with open(file_a, encoding="utf-8") as fa, open(file_b, encoding="utf-8") as fb:
        for line_a, line_b in zip(fa, fb):
            a = clean(line_a)
            b = clean(line_b)

            if not valid_pair(a, b):
                continue

            out_f.write(
                json.dumps(
                    {
                        "text_a": a,
                        "text_b": b,
                        "type": "translation",
                        "lang_a": lang_a,
                        "lang_b": lang_b,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for la, lb in PAIRS:
            merge_pair(la, lb, out_f)

    print(f"\nDone. Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
