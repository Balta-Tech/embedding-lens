"""How many phrases per second does each model embed? Decides which vocabulary sizes are practical.

    uv run python scripts/bench_text.py qwen3vl
"""

import sys
import time

from lens.models import load
from lens.vocab import load as load_vocab

model = load(sys.argv[1])
phrases = load_vocab("medium")[:1024]
model.embed_texts(phrases[:64])  # warm up
for batch_size in (32, 128, 256):
    t0 = time.perf_counter()
    model.embed_texts(phrases, batch_size=batch_size)
    rate = len(phrases) / (time.perf_counter() - t0)
    print(f"{sys.argv[1]} batch {batch_size:>3}: {rate:7.0f} phrases/s -> 119k in {119_000 / rate / 60:5.1f} min")
