# Research notes

Prior work behind the read-out methods, and where each one shows up in this repo. Survey date:
October 2026. Items marked *unverified* come from secondary sources and were not checked against
the paper.

## Modality gap

Image (or video) and text embeddings of contrastive models occupy separate regions of the shared
space: every video vector shares an offset that has nothing to do with its content.

- Liang et al., *Mind the Gap*, NeurIPS 2022 — [arXiv:2203.02053](https://arxiv.org/abs/2203.02053)
- Schrodi et al., *Two Effects, One Trigger*, ICLR 2025 — [arXiv:2404.07983](https://arxiv.org/abs/2404.07983):
  the gap lives in a few dimensions and is driven by information imbalance between modalities.
- Bagad & Zisserman, 2025 — [arXiv:2512.13511](https://arxiv.org/abs/2512.13511): MLLM-based
  embedders, Qwen3-VL included, still show a substantial video–text gap.
- A 2026 study reports that closing the gap can concentrate predictions on a few classes, and that
  CSLS roughly halves this — [arXiv:2609.01103](https://arxiv.org/abs/2609.01103) *(unverified)*.

**Here:** `centered` and `centered_csls` in [`lens/readout.py`](../lens/readout.py) subtract each
modality's mean before comparing; the hubness metric in the experiments watches for the side effect.

## Hubness

In high-dimensional spaces some points ("hubs") are near everything. Generic phrases become hubs
and show up in every read-out.

- Conneau et al., CSLS, 2018 — [arXiv:1710.04087](https://arxiv.org/abs/1710.04087):
  `CSLS(v, t) = 2 cos(v, t) − r_t − r_v`.
- Bogolin et al., QB-Norm, CVPR 2022 — [arXiv:2112.12777](https://arxiv.org/abs/2112.12777):
  inverted softmax over a query bank.
- Wang et al., DBNorm, EMNLP 2023 — [arXiv:2310.11612](https://arxiv.org/abs/2310.11612).
- Chowdhury et al., NNN, 2024 — [arXiv:2410.24114](https://arxiv.org/abs/2410.24114):
  subtract a per-candidate bias `α · mean(top-k similarities to a reference bank)`.

**Here:** `csls`. When ranking phrases for one segment, `r_v` is a constant, so CSLS orders phrases
exactly like NNN with α = 0.5, k = 10. `prior` and `zscore` are the k = all versions
(batch calibration, Zhou et al., ICLR 2024 — [arXiv:2309.17249](https://arxiv.org/abs/2309.17249)).

## Calibration and prompts

- CLIP prompt ensembling (Radford et al., 2021): average a phrase over several templates.
- WaffleCLIP — [arXiv:2306.07282](https://arxiv.org/abs/2306.07282): random descriptors work about
  as well as LLM-written ones, so much of the gain is from averaging.
- Contextual calibration (Zhao et al., 2021 — [arXiv:2102.09690](https://arxiv.org/abs/2102.09690)):
  score a content-free input and remove its bias.

**Here:** [`experiments/templates.py`](../experiments/templates.py) compares templates and an
ensemble. A neutral *text* prompt only shifts all scores of a segment equally (no ranking change);
a neutral *video* (black, grey or noise clip) gives a per-phrase profile, which does change it —
tested as `null` calibration.

## Sparse read-outs

- Bhalla et al., SpLiCE, NeurIPS 2024 — [arXiv:2402.10376](https://arxiv.org/abs/2402.10376):
  centre both modalities, then explain an image embedding as a non-negative sparse sum of concept
  vectors from a 15k-concept LAION vocabulary. Without centring, every metric degrades.
- Gandelsman et al., TextSpan, ICLR 2024 — [arXiv:2310.05916](https://arxiv.org/abs/2310.05916).

**Here:** `sparse_decompose` (greedy matching pursuit with non-negative refits, in the centred
space) and [`experiments/sparse.py`](../experiments/sparse.py).

## Inversion to free text

- vec2text — [arXiv:2310.06816](https://arxiv.org/abs/2310.06816) (trained per encoder).
- ZSInvert — [arXiv:2504.00147](https://arxiv.org/abs/2504.00147): zero-shot, LLM decoding guided
  by embedding similarity.
- DeCap — [arXiv:2303.03032](https://arxiv.org/abs/2303.03032): projects a visual embedding onto a
  memory of text embeddings before decoding; reported zero-shot video captioning.

No inversion method for native *video* embedders was found. Not implemented here; the word
read-out is the training-free end of this spectrum.

## Are video embeddings more than a bag of frames?

- Lei et al., *Single Frame Bias*, ACL 2023 — [arXiv:2206.03428](https://arxiv.org/abs/2206.03428).
- Bagad et al., *Test of Time*, CVPR 2023 — [arXiv:2301.02074](https://arxiv.org/abs/2301.02074):
  video-language models fail at before/after.
- *Seeing the Arrow of Time*, NeurIPS 2025 — [arXiv:2506.03340](https://arxiv.org/abs/2506.03340):
  large multimodal models barely react to reversed video.
- *Chirality in Action*, NeurIPS 2025 — [arXiv:2509.08502](https://arxiv.org/abs/2509.08502):
  open/close-style action pairs are poorly separated by many video embeddings.

**Here:** [`experiments/time_arrow.py`](../experiments/time_arrow.py) — reversal and shuffle
invariance, and a take-off/landing pair that flips meaning when played backwards. SigLIP 2's frame
average is exactly order-invariant, so it is a built-in control.

## Rendered vs real footage

Detecting *generated* images with CLIP features is well studied (UnivFD,
[arXiv:2302.10174](https://arxiv.org/abs/2302.10174); Cozzolino et al.,
[arXiv:2312.00195](https://arxiv.org/abs/2312.00195)). No study of game-engine footage vs real
footage in a video–text embedding space was found.

**Here:** [`experiments/sim_real.py`](../experiments/sim_real.py).

## Vocabularies

| List | Size | Used here |
| --- | --- | --- |
| [RAM++ tags](https://github.com/xinyu1205/recognize-anything) | 4,585 | small |
| ImageNet-21k class names (via WordNet) | ~21.8k | medium |
| Places365 scenes, Kinetics-400 actions | 365 + 400 | medium |
| WordNet 3.0 noun lemmas | ~117k | large |
| [SpLiCE LAION concepts](https://github.com/AI4LIFE-GROUP/SpLiCE) | 15k | not used |
| MetaCLIP metadata | ~500k | not used |
