"""Read a segment embedding back out as words.

All scorers take unit vectors and return one score per phrase, higher = closer. They differ in
how they correct raw cosine similarity:

    raw            cos(v, t)
    neutral        cos(v, t) - cos(v, neutral prompt). Shifts every phrase equally: the ranking is
                   unchanged, only the scale moves. Useful for display, not for choosing words.
    prior          cos(v, t) - mean_j cos(v_j, t). Removes each phrase's general pull towards
                   videos ("hub" phrases that sit close to everything).
    zscore         (cos(v, t) - mu_t) / sigma_t. Like prior, also normalises each phrase's spread.
    csls           2 cos(v, t) - r_t - r_v, with r = mean similarity to the k nearest neighbours
                   of the other modality (Conneau et al. 2018). The standard hubness correction.
    centered       cosine after subtracting each modality's mean vector, which removes the
                   "modality gap" (Liang et al. 2022) shared by all videos / all texts.
    centered_csls  both.
    null           cos(v, t) - mean over null videos (black, grey, noise) of cos(null, t). A
                   per-phrase "content-free" profile, so unlike neutral it does change the ranking.

Everything except raw and neutral needs a reference set of video vectors. Using the corpus being
read out answers "what is distinctive here"; a broad reference set answers "what is in here".
"""

from dataclasses import dataclass

import numpy as np

METHODS = ("raw", "neutral", "null", "prior", "zscore", "csls", "centered", "centered_csls")


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@dataclass
class Phrases:
    """Phrase vectors with their statistics against the reference videos."""

    text: np.ndarray           # (N, d) unit vectors
    centered: np.ndarray       # (N, d) unit vectors after removing the text mean
    mu: np.ndarray             # (N,) mean raw similarity to the reference videos
    sigma: np.ndarray          # (N,) its standard deviation
    r: np.ndarray              # (N,) mean raw similarity to the k nearest reference videos
    r_centered: np.ndarray     # (N,) the same in the centred space
    null: np.ndarray           # (N,) mean raw similarity to the null videos (0 if none given)


class Space:
    def __init__(
        self,
        phrases: list[str],
        text: np.ndarray,
        reference: np.ndarray,
        neutral: np.ndarray | None = None,
        null_videos: np.ndarray | None = None,
        k: int = 10,
    ) -> None:
        self.phrases = phrases
        self.k = min(k, len(reference))
        self.reference = reference
        self.neutral = neutral
        self.null_videos = null_videos
        self.text_mean = text.mean(axis=0)
        self.video_mean = reference.mean(axis=0)
        self.reference_centered = unit(reference - self.video_mean)
        self.vocab = self.describe(text)

    def describe(self, text: np.ndarray) -> Phrases:
        """Statistics for any phrase vectors: the vocabulary, or words a player typed."""
        sims = self.reference @ text.T
        centered = unit(text - self.text_mean)
        sims_c = self.reference_centered @ centered.T
        return Phrases(
            text=text,
            centered=centered,
            mu=sims.mean(axis=0),
            sigma=sims.std(axis=0) + 1e-6,
            r=np.sort(sims, axis=0)[-self.k :].mean(axis=0),
            r_centered=np.sort(sims_c, axis=0)[-self.k :].mean(axis=0),
            null=(self.null_videos @ text.T).mean(axis=0) if self.null_videos is not None
            else np.zeros(len(text)),
        )

    def scores(
        self, v: np.ndarray, method: str = "csls", phrases: Phrases | None = None, alpha: float = 0.5
    ):
        """alpha sets the strength of the CSLS correction: 0.5 is standard CSLS, 0 is none."""
        p = phrases or self.vocab
        if method == "raw":
            return p.text @ v
        if method == "neutral":
            return p.text @ v - float(self.neutral @ v)
        if method == "null":
            return p.text @ v - p.null
        if method == "prior":
            return p.text @ v - p.mu
        if method == "zscore":
            return (p.text @ v - p.mu) / p.sigma
        if method == "csls":
            # r_v always comes from the vocabulary, also when scoring a few extra phrases
            r_v = np.sort(self.vocab.text @ v)[-self.k :].mean()
            return 2 * (p.text @ v - alpha * p.r) - r_v
        vc = unit(v - self.video_mean)
        if method == "centered":
            return p.centered @ vc
        if method == "centered_csls":
            r_v = np.sort(self.vocab.centered @ vc)[-self.k :].mean()
            return 2 * (p.centered @ vc - alpha * p.r_centered) - r_v
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    def top(
        self, v: np.ndarray, n: int = 20, method: str = "csls", alpha: float = 0.5
    ) -> list[tuple[str, float]]:
        s = self.scores(v, method, alpha=alpha)
        idx = np.argpartition(-s, n)[:n]
        idx = idx[np.argsort(-s[idx])]
        return [(self.phrases[i], float(s[i])) for i in idx]

    def rank_of(self, v: np.ndarray, text: np.ndarray, method: str = "csls"):
        """Score phrases that may not be in the vocabulary, and place them in its ranking.

        Returns (scores, ranks, percentiles); rank 1 = would beat every vocabulary phrase.
        """
        vocab_scores = np.sort(self.scores(v, method))
        s = self.scores(v, method, self.describe(text))
        # Tolerance: the same phrase scored alone can differ from the batched score in the last bits
        better = len(vocab_scores) - np.searchsorted(vocab_scores, s + 1e-6, side="left")
        return s, better + 1, 100.0 * (1 - better / len(vocab_scores))


def sparse_decompose(
    v: np.ndarray, space: Space, n_terms: int = 6, candidates: int = 300
) -> tuple[list[tuple[str, float]], float]:
    """Explain v as a short non-negative sum of phrase vectors (greedy matching pursuit).

    Inspired by SpLiCE (Bhalla et al. 2024), but in the centred space. Each step picks the phrase
    that best explains what is left of the vector, so the second word adds something the first
    did not cover instead of repeating it ("snow", "snowfield", "snowy").
    Returns [(phrase, weight)] and the cosine between v and its reconstruction.
    """
    target = unit(v - space.video_mean)
    dictionary = space.vocab.centered
    pool = np.argpartition(-(dictionary @ target), candidates)[:candidates]
    atoms = dictionary[pool]
    residual = target.copy()
    chosen: list[int] = []
    weights = np.zeros(0)
    for _ in range(n_terms):
        gains = atoms @ residual
        gains[chosen] = -np.inf
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        chosen.append(best)
        # Re-fit all chosen weights jointly; clip to keep the explanation additive
        basis = atoms[chosen].T
        weights, *_ = np.linalg.lstsq(basis, target, rcond=None)
        weights = np.clip(weights, 0, None)
        residual = target - basis @ weights
    if not chosen:
        return [], 0.0
    fit = float(unit(atoms[chosen].T @ weights) @ target)
    terms = [(space.phrases[pool[i]], float(w)) for i, w in zip(chosen, weights)]
    return sorted(terms, key=lambda t: -t[1]), fit


def change(
    space: Space, before: np.ndarray, after: np.ndarray, n: int = 6
) -> tuple[float, list[str], list[str]]:
    """What changed between two moments, read out as words.

    The difference between two centred segment vectors is itself a direction in the shared space.
    Phrases along it name what appeared; phrases against it name what went away. Returns
    (1 - cosine between the two moments, appeared, faded).
    """
    a, b = unit(before - space.video_mean), unit(after - space.video_mean)
    amount = 1 - float(a @ b)
    if amount < 1e-4:
        return amount, [], []  # identical moments (e.g. black end frames): nothing to read out
    along = space.vocab.centered @ unit(b - a)
    order = np.argsort(along)
    return (
        amount,
        [space.phrases[i] for i in order[::-1][:n]],
        [space.phrases[i] for i in order[:n]],
    )


def softmax(scores: np.ndarray, temperature: float = 0.02) -> np.ndarray:
    z = (scores - scores.max()) / temperature
    e = np.exp(z)
    return e / e.sum()
