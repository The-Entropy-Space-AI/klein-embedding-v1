# Klein Embedding

A **scratch-trained multilingual sentence embedding model** focused on **semantic similarity and retrieval**, built with clear constraints, transparent methodology, and reproducibility as first principles.

This project prioritizes **understanding, correctness, and inspectability** over leaderboard chasing.

---

## Scope

* **Task**: Sentence similarity & semantic retrieval
* **Model type**: RoBERTa-style encoder (from scratch)
* **Model size**: ~50M parameters
* **Languages**: English + selected Indic languages
* **Max sequence length**: 256
* **Output embeddings**: 128 / 256 / 512 dimensions (via MRL)

This is **not** a general-purpose LLM and **not** optimized for text generation.

---

## Project Philosophy

* Built **from scratch** for learning and full control
* No proprietary weights or hidden dependencies
* Explicit constraints are treated as design inputs
* Failures and limitations are documented, not hidden

The goal is to produce a **solid, usable embedding model** and a **clean reference implementation**.

---

## Repository Structure

```
.
├── data/
│   ├── raw/        # Raw datasets (never modified)
│   ├── processed/  # Cleaned & prepared data
│   └── eval/       # Evaluation-only datasets
│
├── model/          # Model architecture (encoder + heads)
├── train/          # Training logic (tokenizer, datasets, losses)
├── eval/           # Evaluation scripts and benchmarks
├── infer/          # Inference & embedding generation
├── docs/           # Documentation and design notes
│
├── README.md
├── pyproject.toml
└── uv.lock
```

**Separation of concerns is enforced**:

* No code in `data/`
* No training logic in `infer/`
* No evaluation data used during training

---

## Current Status

### Phase 1 — Tokenizer (in progress)

* Scratch-trained BPE tokenizer
* Balanced monolingual corpora (English + Indic)
* Unicode-normalized text
* Frozen after validation

### Upcoming Phases

* Dataset construction (contrastive, QA, translation pairs)
* Model training (MLM warmup → contrastive learning)
* Hard negative mining
* Evaluation & analysis
* Public release

---

## Tokenizer

* **Type**: BPE
* **Library**: Hugging Face `tokenizers`
* **Vocabulary size**: 32,000
* **Normalization**: Unicode NFKC
* **Pre-tokenization**: Whitespace
* **Languages**: Trained on monolingual corpora only

Parallel data is **not** used for tokenizer training.

---

## Training Overview (planned)

1. **Short MLM warmup**

   * Purpose: script grounding & tokenizer adaptation
2. **Contrastive learning**

   * Sentence-level objectives
   * In-batch negatives
3. **Hard negative mining**

   * Iterative refinement
4. **Matryoshka Representation Learning (MRL)**

   * Nested embedding dimensions

---

## Evaluation

Evaluation focuses on:

* Intrinsic similarity behavior
* Cross-lingual alignment
* Retrieval metrics (Recall@k)

Benchmarks are used for **sanity checks and regression detection**, not leaderboard optimization.

---

## What This Project Is NOT

* Not claiming state-of-the-art results
* Not trained on proprietary data
* Not optimized for generation
* Not benchmark-chasing

Any performance claims will be **scoped and qualified**.

---

## Reproducibility

* All datasets are public
* All preprocessing steps are scripted
* Tokenizer and model configs are versioned
* Training runs are deterministic where possible

---

## License

License information will be finalized upon first public model release.

---

## Status & Disclaimer

This project is under active development.
Interfaces, configs, and results may change as understanding improves.

If you use this work, **assume evolving behavior**, not a frozen product.

---

## Contact

Issues and discussions are preferred over private messages.
Feedback is welcome if it is **concrete and technically grounded**.
