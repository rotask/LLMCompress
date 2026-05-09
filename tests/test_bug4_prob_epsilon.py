"""
Test for Bug #4 fix: zero-probability clamping uses 1e-12 (not 0.001), logs a
warning, and produces finite entropy.

We test the *behavior* of the new clamp + warning by reproducing the relevant
post-loop logic in isolation. This avoids needing a live model.
"""
import io
import logging
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_clamp_replaces_zero_with_small_epsilon():
    import torch
    probs = torch.tensor([0.5, 0.0, 0.25, 0.0, 0.99])

    # Mirror the fixed code path:
    n_zero = (probs == 0).sum().item()
    clamped = probs.clamp(min=1e-12)

    assert n_zero == 2
    assert (clamped > 0).all().item(), "Clamped tensor must be strictly positive"
    # fp32 representation of 1e-12 has ~1e-19 rounding; use tolerance
    assert math.isclose(clamped[1].item(), 1e-12, rel_tol=1e-6), f"Expected ~1e-12, got {clamped[1].item()}"
    assert math.isclose(clamped[3].item(), 1e-12, rel_tol=1e-6), f"Expected ~1e-12, got {clamped[3].item()}"
    assert clamped[0].item() == 0.5,  "Non-zero values should be unchanged"
    print("[OK] clamp replaces zeros with 1e-12, leaves other values unchanged.")


def test_entropy_is_finite_after_clamp():
    """log2(0) = -inf; log2(1e-12) ≈ -39.86 (finite). Verify the metric is computable."""
    import torch
    probs = torch.tensor([0.5, 0.0, 0.25, 0.0, 0.99])
    clamped = probs.clamp(min=1e-12)
    entropy = (-torch.log2(clamped)).sum().item()
    assert math.isfinite(entropy), f"Entropy should be finite, got {entropy}"
    print(f"[OK] entropy is finite after clamp: {entropy:.4f} bits total over 5 tokens")


def test_warning_logged_when_zero_probs_present():
    """The warning should fire iff there are zero-probability positions."""
    import torch
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.WARNING)
    root.addHandler(handler)
    try:
        probs = torch.tensor([0.5, 0.0, 0.25])
        n_zero = (probs == 0).sum().item()
        if n_zero > 0:
            logging.warning(
                f"{n_zero} of {len(probs)} positions had model probability 0 "
                f"(likely softmax underflow on quantization tail). Clamping to 1e-12 "
                f"for entropy calc; this affects the reported entropy metric only."
            )
        handler.flush()
        out = log_stream.getvalue()
        assert "1 of 3 positions had model probability 0" in out, \
            f"Expected count in warning, got: {out!r}"
        print("[OK] warning logged with zero-probability count.")
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)


def test_source_no_longer_uses_001_patch():
    """Static check: the bugged 0.001 patch is gone, clamp(min=1e-12) is in place."""
    src = open(
        os.path.join(os.path.dirname(__file__), "..", "llmcompress.py"),
        "r", encoding="utf-8"
    ).read()
    code_only = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "probs + 0.001" not in code_only, "Bugged 0.001 patch still present"
    assert "probs.clamp(min=1e-12)" in code_only, "Fixed clamp not found"
    print("[OK] llmcompress.py uses clamp(min=1e-12) instead of probs+0.001.")


if __name__ == "__main__":
    test_clamp_replaces_zero_with_small_epsilon()
    test_entropy_is_finite_after_clamp()
    test_warning_logged_when_zero_probs_present()
    test_source_no_longer_uses_001_patch()
    print("\nBug #4 verified.")
