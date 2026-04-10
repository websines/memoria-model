#!/usr/bin/env python3
"""Warm the HuggingFace streaming cache before training.

Downloads the first shard of each dataset so the training loop
hits local cache instead of doing first-time downloads under
DDP timeout pressure.

Run once before training:
    python scripts/warmup_cache.py

Takes ~5-10 minutes depending on network. Uses ~2-5 GB disk.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memoria.data.curated import CURATED_SOURCES, _load_hf_stream


def main():
    ok = 0
    fail = 0
    skipped = []

    print(f"Warming cache for {len(CURATED_SOURCES)} datasets...")
    print(f"Cache dir: {os.environ.get('HF_HOME', '~/.cache/huggingface')}\n")

    for source in CURATED_SOURCES:
        try:
            stream = _load_hf_stream(source)
            # Pull 5 samples to warm the cache (downloads first parquet shard)
            for _ in range(5):
                next(stream)
            del stream
            print(f"  ✓ {source.name}")
            ok += 1
        except Exception as e:
            err = str(e)[:80]
            print(f"  ✗ {source.name}: {err}")
            skipped.append(source.name)
            fail += 1

    print(f"\nDone: {ok} cached, {fail} failed")
    if skipped:
        print(f"Skipped (weight redistributed): {', '.join(skipped)}")
    print("\nYou can now run training — data loads from cache.")


if __name__ == "__main__":
    main()
