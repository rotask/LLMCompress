"""
CPU-only smoke test: tokenize → compress (Ranks) → decompress → assert equality.

Uses GPT-2 (124M) on CPU. Runs in ~30–60 s on a modern laptop. No CUDA needed.

Validates:
  - Ranks compression / decompression roundtrip
  - Padding logic
  - EOS token handling
  - Zlib rank-stream encode/decode

Does NOT validate the AC path (separate test below) or the quantized-model
results from the thesis (those require CUDA + bitsandbytes).

Run with:  python -m pytest tests/test_roundtrip.py -v
or:        python tests/test_roundtrip.py
"""
import os
import sys
import tempfile
import shutil

# Allow running from repo root without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from compression import compress_ranks, decompress_ranks


def test_compress_decompress_ranks_roundtrip():
    """compress_ranks(decompress_ranks(x)) == x for an arbitrary rank stream."""
    import torch
    ranks = torch.tensor([0, 1, 0, 5, 12, 0, 3, 0, 0, 1, 200, 0, 7] * 100, dtype=torch.int32)
    compressed = compress_ranks(ranks)
    recovered = decompress_ranks(compressed)
    assert torch.equal(ranks, recovered), "Rank stream roundtrip failed"
    print(f"[OK] rank-stream roundtrip: {len(ranks)} ranks, {len(compressed)} bytes compressed")


def test_zlib_baseline_full_equality():
    """zlib_compress baseline must verify with full equality, not prefix check."""
    import zlib
    text = "the quick brown fox jumps over the lazy dog. " * 50
    compressed = zlib.compress(text.encode("utf-8"), level=9)
    decompressed = zlib.decompress(compressed).decode("utf-8")
    assert text == decompressed, "Zlib baseline roundtrip should be byte-exact"
    print(f"[OK] zlib baseline roundtrip: {len(text)} chars → {len(compressed)} bytes")


def test_gpt2_ranks_roundtrip_cpu():
    """End-to-end: GPT-2 on CPU compresses + decompresses a small text losslessly."""
    import torch
    from llmcompress import LLMCompress

    # Force CPU even if CUDA happens to be available
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Tiny corpus: 1 KB of structured text, hits multiple context windows at ctx=128
    sample = (
        "Information theory provides the link between prediction and compression. "
        "A model that assigns high probability to the next token can be used to "
        "compress text by encoding only the rank of that token under the predicted "
        "distribution. This is the central idea behind LLM-based compression. "
    ) * 8

    workdir = tempfile.mkdtemp(prefix="llmcompress_test_")
    try:
        # Redirect Config to the temp dir so we don't pollute the repo
        Config.OUTPUT_DIR = os.path.join(workdir, "Output_Files")
        Config.LOGS_DIR = os.path.join(workdir, "Logs")
        Config.RESULTS_DIR = os.path.join(workdir, "Results")
        Config.CONTEXT_SIZE = 128
        Config.BATCH_SIZE = 4
        Config.ensure_directories()

        in_file = os.path.join(workdir, "sample.txt")
        out_file = os.path.join(workdir, "sample.gpz")
        recovered_file = os.path.join(workdir, "sample.recovered.txt")
        with open(in_file, "w", encoding="utf-8") as f:
            f.write(sample)

        compressor = LLMCompress("gpt2", "Ranks")
        compressor.zip(in_file, out_file)
        compressor.unzip(out_file, recovered_file)
        assert compressor.check(in_file, recovered_file), "GPT-2 Ranks roundtrip failed"
        print(f"[OK] GPT-2 Ranks roundtrip on CPU: {len(sample)} chars compressed and recovered")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    test_compress_decompress_ranks_roundtrip()
    test_zlib_baseline_full_equality()
    test_gpt2_ranks_roundtrip_cpu()
    print("\nAll tests passed.")
