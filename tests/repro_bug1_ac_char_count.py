"""
Phase 1 reproduction for Bug #1:
Arithmetic_Coder.py:76 computes:
    num_characters = len(tokenizer.batch_decode(tokens, skip_special_tokens=True))

`tokens` is a flat list[int] from `tokenizer.encode(...)`.
`tokenizer.batch_decode(tokens, ...)` treats each int as its own batch element,
returning a list of N strings (one per token, decoded with no surrounding
context). `len(...)` returns N — the token count, not the character count.

The metric is then named `num_characters` and used to compute entropy
(`bpc = entropy_total / num_characters`) and compression ratio
(`8 * filesize / num_characters`). Both metrics become per-token rather than
per-character — wrong by a factor of ~5 for English text (chars/token ≈ 5).
"""
from transformers import GPT2Tokenizer

text = "Information theory is the link between prediction and compression."
print(f"Text:                                {text!r}")
print(f"len(text)                            = {len(text)}  ← TRUE character count")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode(text)
print(f"len(tokens)                          = {len(tokens)}  ← token count")

# What the bugged code computes:
bugged = len(tokenizer.batch_decode(tokens, skip_special_tokens=True))
print(f"BUGGED  len(batch_decode(tokens))    = {bugged}  ← list length = token count")

# Show what batch_decode actually returned (first few items)
batch_strs = tokenizer.batch_decode(tokens, skip_special_tokens=True)
print(f"        batch_decode returns        = {batch_strs}")

# Correct version: decode the full sequence as a single text and take its length
correct = len(tokenizer.decode(tokens, skip_special_tokens=True))
print(f"CORRECT len(decode(tokens))          = {correct}")

print()
assert bugged == len(tokens),  f"Reproduction failed: bugged ({bugged}) != len(tokens) ({len(tokens)})"
assert correct == len(text),   f"Reproduction failed: correct ({correct}) != len(text) ({len(text)})"
assert bugged != correct,      f"Reproduction failed: bugged equals correct"
print(f"[REPRODUCED] Bug #1 confirmed.")
print(f"  num_characters reported as {bugged} (token count) instead of {correct} (character count).")
print(f"  Distortion factor on this sample: {correct / bugged:.2f}x")
