"""
Phase 1 reproduction for Bug #5:
zlib_compress.py:85 uses `data.startswith(decompressed_message[:200])` as the
verification check. This is wrong for two reasons:

  1. A lossless compressor must produce a byte-exact roundtrip. The original
     and decompressed must be EQUAL, not "original begins with decompressed
     prefix".
  2. .startswith only checks the first 200 chars. Any divergence after byte
     200 is silently accepted as success.

This script demonstrates the prefix-check passing on a deliberately tampered
'decompressed' value that does NOT equal the original.
"""

original = "A" * 200 + "BBBB"  # 204 chars
tampered_decompressed = "A" * 200 + "ZZZZ"  # First 200 chars match, rest differs

# What the bugged code does:
bugged_passes = original.startswith(tampered_decompressed[:200])

# What a correct check would do:
correct_passes = original == tampered_decompressed

print(f"original                  = {original!r}")
print(f"tampered_decompressed     = {tampered_decompressed!r}")
print(f"bugged check passes?      = {bugged_passes}")
print(f"correct check passes?     = {correct_passes}")
print()
assert bugged_passes is True,  "Reproduction failed: bugged check unexpectedly rejected"
assert correct_passes is False, "Reproduction failed: correct check unexpectedly accepted"
print("[REPRODUCED] Bug #5 confirmed: prefix check accepts non-equal data.")
