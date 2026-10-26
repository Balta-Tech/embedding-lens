"""How slow is it? Embedding rates measured while indexing, and what they mean per vocabulary size.

Reads the progress lines lens.index writes (data/logs/index.log). Rates are averages over the whole
run on an Apple M3 Pro, sometimes with other jobs sharing the GPU, so treat them as ballpark.

    uv run python -m experiments.speed
"""

import re
from collections import defaultdict

from experiments.common import save, table
from lens.config import DATA
from lens.vocab import load as load_vocab

LOG = DATA / "logs" / "index.log"


def main() -> None:
    text = LOG.read_text().replace("\r", "\n")
    vocab_rates = defaultdict(list)
    segment_rate = {}
    for line in text.splitlines():
        m = re.match(r"(\w+) vocab @\d+: \d+/\d+\s+([\d.]+)/s", line)
        if m:
            vocab_rates[m[1]].append(float(m[2]))
        m = re.match(r"(\w+) segments: (\d+)/\2\s+([\d.]+)/s", line)
        if m:
            segment_rate[m[1]] = float(m[3])
    sizes = {tier: len(load_vocab(tier)) for tier in ("small", "medium", "large")}
    rows = []
    for model, rates in vocab_rates.items():
        rate = sum(rates) / len(rates)
        row = {"model": model, "phrases_per_s": rate, "segments_per_s": segment_rate.get(model, 0.0)}
        row |= {f"{tier}_min": n / rate / 60 for tier, n in sizes.items()}
        rows.append(row)
    save("speed", {"vocab_sizes": sizes, "rows": rows})
    print(table(rows, list(rows[0])))


if __name__ == "__main__":
    main()
