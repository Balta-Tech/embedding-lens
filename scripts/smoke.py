"""Sanity check: do the right words win for three very different segments?

    uv run python scripts/smoke.py qwen3vl
"""

import sys
import time

from lens.models import load
from lens.videos import SOURCES_BY_ID, segments

WORDS = ["mountain", "snow", "airplane cockpit", "video game", "race car", "computer screen"]
PICKS = [("zugspitze", 40), ("f16", 12), ("trex", 10)]

model_name = sys.argv[1]
t0 = time.perf_counter()
model = load(model_name)
print(f"load {time.perf_counter() - t0:.1f}s")

segs = [segments(SOURCES_BY_ID[v])[i] for v, i in PICKS]
t0 = time.perf_counter()
words = model.embed_texts(WORDS)
print(f"{len(WORDS)} texts {time.perf_counter() - t0:.2f}s")
t0 = time.perf_counter()
vids = model.embed_segments(segs)
print(f"{len(segs)} segments {time.perf_counter() - t0:.2f}s")

sims = vids @ words.T
print(f"{'':>14}" + "".join(f"{w[:12]:>14}" for w in WORDS))
for (v, i), row in zip(PICKS, sims):
    print(f"{v + ':' + str(i):>14}" + "".join(f"{s:>14.3f}" for s in row))
