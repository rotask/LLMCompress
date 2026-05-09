"""
Test for Bug #5 fix: zlib_compress.py uses full equality check.

We can't easily test the script's main() (it has hardcoded paths and prints).
Instead we statically inspect the relevant logic to confirm:
  - The fixed code uses `data == decompressed_message`
  - The fixed code does NOT use `.startswith(...[:200])`
"""
import os
import sys
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_zlib_check_uses_full_equality():
    src = open(os.path.join(REPO_ROOT, "zlib_compress.py"), "r", encoding="utf-8").read()
    assert "data == decompressed_message" in src, "Fixed equality check not found"
    assert "decompressed_message[:200]" not in src, "Bugged prefix check still present"
    assert ".startswith(" not in src or "decompressed_message[:200]" not in src, \
        "Prefix-check pattern still present"
    print("[OK] zlib_compress.py uses full equality check.")


if __name__ == "__main__":
    test_zlib_check_uses_full_equality()
    print("\nBug #5 verified: prefix-check replaced with full equality.")
