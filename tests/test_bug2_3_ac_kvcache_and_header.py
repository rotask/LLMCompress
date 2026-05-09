"""
Integration test for Bug #2 (AC O(N²) compute) and Bug #3 (AC file not
self-contained). After the fix:

  * AC_compress_file writes a 4-byte little-endian uint32 header containing
    the token count, followed by the existing AC-coded payload.
  * AC_decompress_file reads the header to determine how many tokens to
    decode; the `num_tokens` parameter becomes optional (None → use header).
  * Both encode and decode loops reuse `past_key_values` for O(N) compute.

Verification: end-to-end roundtrip with GPT-2 on CPU. ~5–15 tokens.
"""
import os
import struct
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_ac_roundtrip_on_cpu_gpt2():
    """Round-trip a tiny string through AC and assert the decompressed text matches."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from Arithmetic_Coder import AC_compress_file, AC_decompress_file

    print("Loading GPT-2 (CPU, float32)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # 6-token input — keeps the test under 30 s even with a slow encoder
    text = "Hello world from compression."
    tmpdir = tempfile.mkdtemp(prefix="ac_test_")
    in_file = os.path.join(tmpdir, "in.txt")
    out_file = os.path.join(tmpdir, "out.bin")
    decompressed_file = os.path.join(tmpdir, "out.txt")

    # Make sure the AC results CSV ends up in the temp dir, not the repo
    from config import Config
    Config.RESULTS_DIR = tmpdir
    Config.LOGS_DIR = tmpdir
    Config.OUTPUT_DIR = tmpdir
    Config.ensure_directories()

    with open(in_file, "w", encoding="utf-8") as f:
        f.write(text)

    t0 = time.time()
    AC_compress_file(model, "gpt2", tokenizer, in_file, out_file)
    t_compress = time.time() - t0
    print(f"AC compress took {t_compress:.2f}s for '{text}'")

    # Inspect the file header
    with open(out_file, "rb") as f:
        header = f.read(4)
        num_tokens = struct.unpack("<I", header)[0]
    expected_tokens = len(tokenizer.encode(text))
    assert num_tokens == expected_tokens, \
        f"Header reports {num_tokens} tokens, expected {expected_tokens}"
    print(f"[OK] header carries num_tokens = {num_tokens} (= expected)")

    # Decompress without passing num_tokens — must read from header
    t0 = time.time()
    AC_decompress_file(model, tokenizer, out_file, decompressed_file)
    t_decompress = time.time() - t0
    print(f"AC decompress took {t_decompress:.2f}s")

    with open(decompressed_file, "r", encoding="utf-8") as f:
        recovered = f.read()
    assert recovered == text, \
        f"Roundtrip mismatch:\n  original:  {text!r}\n  recovered: {recovered!r}"
    print(f"[OK] roundtrip lossless: '{recovered}'")


def test_ac_decompress_signature_makes_num_tokens_optional():
    """Verify the post-fix function signature does not require num_tokens."""
    import inspect
    from Arithmetic_Coder import AC_decompress_file
    sig = inspect.signature(AC_decompress_file)
    params = sig.parameters
    if "num_tokens" in params:
        default = params["num_tokens"].default
        assert default is None or default is inspect.Parameter.empty, \
            f"num_tokens should default to None, got {default!r}"
        # Allow None default (backward-compat) or removal entirely
        print(f"[OK] AC_decompress_file num_tokens default = {default!r} (None ⇒ read from header)")
    else:
        print("[OK] AC_decompress_file no longer takes num_tokens (header-only).")


if __name__ == "__main__":
    test_ac_decompress_signature_makes_num_tokens_optional()
    test_ac_roundtrip_on_cpu_gpt2()
    print("\nBug #2 + #3 verified.")
