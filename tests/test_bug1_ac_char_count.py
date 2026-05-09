"""
Test for Bug #1 fix: AC char-count uses tokenizer.decode (singular, joined),
not tokenizer.batch_decode (returns a list).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_decode_yields_correct_character_count():
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    text = "Information theory is the link between prediction and compression."
    tokens = tokenizer.encode(text)

    # The fixed approach (what AC code now uses):
    num_characters = len(tokenizer.decode(tokens, skip_special_tokens=True))

    assert num_characters == len(text), \
        f"Expected {len(text)}, got {num_characters}"
    print(f"[OK] tokenizer.decode(tokens) yields {num_characters} chars (matches len(text))")


def test_ac_source_uses_decode_not_batch_decode():
    """Static check: the bugged pattern is gone (excluding comments), fixed pattern is in place."""
    src = open(
        os.path.join(os.path.dirname(__file__), "..", "Arithmetic_Coder.py"),
        "r", encoding="utf-8"
    ).read()
    # Strip comments before scanning so explanatory text doesn't trigger false positives.
    code_only = "\n".join(
        line for line in src.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "len(tokenizer.batch_decode(tokens" not in code_only, \
        "Bugged batch_decode pattern still present in Arithmetic_Coder.py"
    assert "len(tokenizer.decode(tokens" in code_only, \
        "Fixed decode pattern not found in Arithmetic_Coder.py"
    print("[OK] Arithmetic_Coder.py: bugged batch_decode pattern replaced with decode.")


if __name__ == "__main__":
    test_decode_yields_correct_character_count()
    test_ac_source_uses_decode_not_batch_decode()
    print("\nBug #1 verified.")
