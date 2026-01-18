# Klein Embedding v1

A **scratch-trained multilingual sentence embedding model** focused on **semantic similarity and retrieval**. This model is built with a focus on transparency, efficiency, and reproducibility.

## Model Overview

* **Model Type**: RoBERTa-style encoder
* **Parameter Count**: **35.05 Million**
* **Hidden Dimension**: **480**
* **Layers**: **7**
* **Max Sequence Length**: **128 tokens**
* **Vocabulary Size**: **32,000**
* **License**: Apache-2.0

---

## Evaluation Results

The following metrics represent the model's performance on standard Semantic Textual Similarity (STS) benchmarks:

| Dataset | Spearman | Pearson | Samples |
| --- | --- | --- | --- |
| **STSb** | 40.54% | 39.64% | 1,379 |
| **SICK-R** | 51.69% | 51.78% | 9,927 |
| **STS12** | 42.59% | 36.88% | 3,108 |
| **STS13** | 37.76% | 37.99% | 1,500 |
| **STS14** | 36.99% | 36.55% | 3,750 |
| **STS15** | 52.29% | 53.14% | 3,000 |
| **STS16** | 50.35% | 49.56% | 1,186 |
| **Average** | **44.60%** | **43.65%** | — |

---

## Technical Methodology

* **Architecture**: Optimized RoBERTa-base with a reduced hidden dimension (480) and depth (7 layers) for high-speed inference without sacrificing semantic depth.
* **Tokenizer**: Custom BPE tokenizer trained from scratch on balanced monolingual English and Indic corpora.
* **Training Objective**: Contrastive learning using in-batch negatives and alignment loss to map similar sentences to a shared vector space.
* **Format**: Distributed in **Safetensors** for secure and fast loading.

---

## Quick Start (Usage)

### Using Transformers

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_id = "the-entropy-space-ai/klein-embedding-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

text = "Your sentence here"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    # Mean pooling to get a single 480-dimension vector
    embeddings = outputs.last_hidden_state.mean(dim=1)

print(embeddings.shape) # torch.Size([1, 480])

```

---

## Project Philosophy

* **Efficiency First**: At ~133MB, this model is designed to run on standard CPUs with very low latency (~78ms per sentence).
* **Full Control**: Every step, from tokenizer normalization to the final contrastive loss, is documented and reproducible.
* **Transparent Limitations**: This model is optimized for sentence similarity and retrieval, not for text generation or general-purpose LLM tasks.

Would you like me to create a "Similarity Example" code block for the README so users can see how to compare two sentences?