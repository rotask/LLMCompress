# LLMCompress — LLM-Enhanced Lossless Text Compression

> MSc thesis · Imperial College London · Department of Electrical and Electronic Engineering · 2024
> Supervised by Prof. Deniz Gündüz

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-RTX%203090-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey)

Lossless text compression that uses next-token probabilities from large language models in place of a static probabilistic model. The idea exploits the duality between prediction and compression rooted in information theory: a better predictor is a better compressor.

> **Headline result:** Yi-34B with a 1024-token context achieves **1.135 bits/char on `text8`** — a **57% reduction vs Zlib (2.64 bpc)** and a 39% reduction vs the GPT-2 baseline (1.865 bpc).

---

## Background — why this works

Shannon's source coding theorem gives the entropy of a source as the theoretical lower bound on the average code length per symbol. The cross-entropy of a model on a corpus is the average code length you'd achieve if you used that model's predicted probabilities to drive an arithmetic coder. So **the better an LLM predicts the next token, the closer it pushes you to the source's true entropy.**

LLMCompress tests this empirically across 8 modern LLMs, 4 context sizes, and 2 datasets, integrating model probabilities with Zlib via a rank-based encoding scheme.

---

## Results

### Compression on text8 (1 MB Wikipedia subset), context size 1024

Sorted by compression ratio (lower = better). All values in bits per character.

| Model           | Params  | Tokens (NT) | Chars/Tok | Entropy H | Ratio ρ   | Redundancy |
| --------------- | ------- | ----------- | --------- | --------- | --------- | ---------- |
| **Yi-34B**      | 34 B    | 207,783     | 5.05      | **0.632** | **1.135** | 0.800      |
| Mixtral-8x7B    | 47 B*   | 219,000     | 4.79      | 0.626     | 1.145     | 0.799      |
| Llama-3         | 8 B     | 198,593     | 5.28      | 0.739     | 1.266     | 0.769      |
| Mistral-7B      | 7 B     | 218,999     | 4.79      | 0.726     | 1.280     | 0.767      |
| Llama-2         | 7 B     | 225,898     | 4.64      | 0.715     | 1.283     | 0.778      |
| Llama-3.1       | 8 B     | 198,593     | 5.28      | 0.789     | 1.330     | 0.754      |
| NeMo            | 12 B    | 201,445     | 5.21      | 0.854     | 1.436     | 0.738      |
| GPT-2 (baseline)| 124 M   | 203,334     | 5.16      | 1.185     | 1.865     | 0.608      |
| **Zlib (DEFLATE)** | —    | —           | —         | —         | **2.64**  | —          |

\* Mixtral-8x7B has 47 B total parameters with ~13 B active per token (sparse mixture-of-experts).

### Key findings

1. **Larger models compress better.** Yi-34B and Mixtral-8x7B are the top performers across all four context sizes (128, 256, 512, 1024).
2. **Context length matters.** Yi-34B drops from 1.374 bpc at ctx=128 to 1.135 bpc at ctx=1024 — a 17% improvement from context alone.
3. **Tokenization matters.** Models with byte-fallback BPE (Llama-2, Mistral-7B) maintain consistent tokenization on rare strings; models with finer chars/token ratios (Llama-3) achieve lower entropy per character.
4. **Compression vs speed is a real trade-off.** Mixtral-8x7B takes ~65 min to compress 1 MB on an RTX 3090; Llama-3-8B is ~3× faster at ~5% worse compression. For most workloads Llama-3 is the practical sweet spot.

### Side-information experiment: prepending a 150-word summary

| Context | Δ compression ratio (with summary vs without) |
| ------- | --------------------------------------------- |
| 128     | **−5.43%** (helps)                            |
| 256     | −3.29% (helps)                                |
| 512     | +3.80% (hurts)                                |
| 1024    | +3.58% (hurts)                                |

Short context windows benefit from a prepended global summary because the model otherwise can't "see" the document's overall structure. At larger context sizes the model already has enough text in-window, and the summary becomes redundant overhead.

A more ambitious **Enhanced Contextual Compression** scheme that prepends per-window summaries was implemented but did not achieve full lossless reconstruction during decompression (token-rank misalignment). Documented in §5.7 of the thesis as a negative result for future work.

---

## Reproducibility note

All non-GPT-2 results in the table above were obtained with **4-bit NF4 quantization** (bitsandbytes, `bnb_4bit_compute_dtype=bfloat16`, double-quant on) on **2× NVIDIA RTX 3090 (24 GB each)** with Flash-Attention 2 where supported. GPT-2 was run unquantized at full precision.

Quantization was a hard requirement to fit Yi-34B and Mixtral-8x7B in 48 GB VRAM at context size 1024. Reported entropies and compression ratios are therefore quantized-model performance. Full-precision results would likely be marginally better but were not measured. See thesis §3.2 for the experimental setup and §6.1 for hardware/environment details.

---

## Known issues (post-thesis review)

A code review after submission identified the following issues in the codebase. They do not affect the canonical Ranks-method results reported above (text8 / BookCorpus, all 8 models, all context sizes), which were validated by the round-trip equality check in `LLMCompress.check`. They are flagged here for transparency and for anyone reusing the code.

- **Arithmetic-Coding path (`Arithmetic_Coder.py`)**: (1) the character-count metric uses `len(tokenizer.batch_decode(...))` which returns the list length, not the character count — so AC entropy/ratio metrics, where reported, are inconsistent with the Ranks-method definitions; (2) the encoder loop does not reuse `past_key_values`, making AC O(N²) in compute; (3) the compressed `.bin` file does not embed the original token count, so it is not self-contained for decompression. The Ranks method (canonical for this thesis) does not share these issues.
- **Zero-probability epsilon (`llmcompress.py`)**: tokens with `prob == 0` (rare, but possible at the quantization tail) are silently shifted to `0.001`. This affects the `entropy` metric only when triggered; the rank stream and the round-trip-verified compression ratio are unaffected.
- **`zlib_compress.py` round-trip check** uses `data.startswith(decompressed[:200])` — a prefix-only sanity check, not a true equality assertion. The Zlib baseline numbers in the thesis come from `os.path.getsize` of the compressed file, which is independent of this check.

---

## Method — rank-based encoding with Zlib

For each token position, the LLM produces a probability distribution over the vocabulary. The token actually present in the source text is converted to its **rank** under that distribution (rank 0 = most likely, rank 1 = second-most likely, …). The sequence of ranks is then encoded with Zlib (DEFLATE).

This works because for a well-calibrated LLM most tokens have rank 0–3, producing a heavily skewed distribution that Zlib compresses near-optimally. Arithmetic coding directly on the LLM probabilities is also implemented (`Arithmetic_Coder.py`) as a theoretical reference point.

Full mathematical formulation and architectural details: §2.9 and §3 of the thesis.

---

## Models supported

| Family   | Variants tested                          | Tokenizer            |
| -------- | ---------------------------------------- | -------------------- |
| GPT      | GPT-2 (baseline)                         | BPE                  |
| Llama    | Llama-2, Llama-3, Llama-3.1              | Byte-fallback BPE    |
| Mistral  | Mistral-7B, Mixtral-8x7B (MoE)           | Byte-fallback BPE    |
| Yi       | Yi-34B                                   | BPE                  |
| NeMo     | NeMo (Mistral-NeMo-12B)                  | Tekken               |

---

## Quick start

### Prerequisites

- NVIDIA GPU(s) — **2× RTX 3090 recommended for ≥7B models**; ≥48 GB VRAM total to run Yi-34B / Mixtral-8x7B at ctx=1024
- Linux (Ubuntu 20.04+ tested)
- Python 3.9+
- CUDA 11.8+ and PyTorch with CUDA support
- Hugging Face access tokens for gated models (Llama, Mistral families)

### Installation

```bash
git clone https://github.com/rotask/LLMCompress.git
cd LLMCompress
conda env create -f environment.yml
conda activate llmzip-env
```

### Run an experiment

```bash
python main.py \
  --model gpt2 \
  --compression_method Ranks \
  --input_file "Data/text8_1MB.txt" \
  --batch_size 32 \
  --context_size 512
```

```bash
python main.py --help   # all options
```

---

## Project structure

```
LLMCompress/
├── main.py                # Entry point: parses args, runs experiment
├── llmcompress.py         # Core LLM-driven compression / decompression loop
├── compression.py         # Encoding pipeline (ranks, AC integration)
├── Arithmetic_Coder.py    # Standalone arithmetic-coder reference
├── zlib_compress.py       # Zlib baseline + rank-stream encoding
├── models.py              # Model + tokenizer loading (HF transformers)
├── config.py              # Hyperparameters and run config
├── logger.py              # Per-run logging to disk
├── environment.yml        # Conda spec
├── Data/                  # Input corpora (text8, BookCorpus subsets)
├── Output_Files/          # Compressed artifacts
├── Logs/                  # Per-run logs
└── Results/               # Aggregated metrics CSVs
```

---

## Citation

If you reference this work, please cite the thesis:

```bibtex
@mastersthesis{rotas2024llmcompress,
  author  = {Rotas, Konstantinos},
  title   = {Large Language Models for Data Compression},
  school  = {Imperial College London},
  year    = {2024},
  type    = {{MSc} thesis},
  address = {Department of Electrical and Electronic Engineering}
}
```

---

## Acknowledgments

Master's thesis at Imperial College London, supervised by Prof. Deniz Gündüz. Special thanks to Selim Yilmaz for technical discussions, and to **EUROTANKERS INC** and the **Union of Greek Shipowners** for the scholarship that supported this work.

## License

Released under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](LICENSE). You may share with attribution; commercial use, modification, and derivative works are not permitted.
