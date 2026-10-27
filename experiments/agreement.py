"""Experiment 10: which automatic metric agrees with judging by eye?

Joins the judged precision (eyeball_score.json) with the automatic metrics (methods.json) over
every model, read-out and vocabulary size that both cover, and reports Spearman rank correlations.

    uv run python -m experiments.agreement
"""

import json

import numpy as np

from experiments.common import save
from lens.config import RESULTS_DIR

METRICS = {
    "second opinion": "judge",
    "round trip (recall@5)": "r@5",
    "round trip (recall@1)": "r@1",
    "word variety (coverage)": "coverage",
}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        return np.argsort(np.argsort(v))

    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def main() -> None:
    methods = json.loads((RESULTS_DIR / "methods.json").read_text())
    judged = json.loads((RESULTS_DIR / "eyeball_score.json").read_text())
    index = {(r["model"], r["vocab"], r["method"]): r for r in methods}
    rows = []
    for r in judged:
        method, _, tier = r["method"].partition("/")
        auto = index.get((r["model"], tier or "medium", method))
        if auto:
            rows.append((r["precision@5"], *(auto[k] for k in METRICS.values())))
    data = np.array(rows)
    result = {
        "configurations": len(rows),
        "correlations": {name: spearman(data[:, 0], data[:, i + 1]) for i, name in enumerate(METRICS)},
    }
    save("agreement", result)
    print(f"{len(rows)} configurations")
    for name, rho in result["correlations"].items():
        print(f"  {name:>26}: {rho:+.2f}")


if __name__ == "__main__":
    main()
