"""Score the side-by-side read-outs against the judgements in judgements.json.

Precision@5: the share of each read-out's top 5 words judged correct, per model and method.

    uv run python -m experiments.eyeball_score            # needs data/results/eyeball.json
    uv run python -m experiments.eyeball_score --review   # list words not marked correct
"""

import json
import sys
from pathlib import Path

from experiments.common import save, table
from lens.config import RESULTS_DIR

JUDGEMENTS = json.loads((Path(__file__).parent / "judgements.json").read_text())


def main() -> None:
    panels = json.loads((RESULTS_DIR / "eyeball.json").read_text())
    if "--review" in sys.argv:
        for key in next(iter(panels.values())):
            wrong = sorted({w for m in panels.values() for ws in m[key].values() for w in ws}
                           - set(JUDGEMENTS[key]))
            print(f"{key}: {'; '.join(wrong)}")
        return
    rows = []
    for model, segments in panels.items():
        methods = next(iter(segments.values())).keys()
        for method in methods:
            hits = [w in JUDGEMENTS[key] for key, rows_ in segments.items() for w in rows_[method]]
            first = [rows_[method][0] in JUDGEMENTS[key] for key, rows_ in segments.items()]
            rows.append({"model": model, "method": method,
                         "precision@5": sum(hits) / len(hits), "top1": sum(first) / len(first)})
    save("eyeball_score", rows)
    print(table(rows, list(rows[0])))


if __name__ == "__main__":
    main()
