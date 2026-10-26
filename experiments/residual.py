"""Experiment 7: what does the video model see that its single frames don't?

For each sampled segment, embed its frames one by one as still images with the *same* model and
average them. The difference between the native video vector and that frame average is the part
of the embedding that only exists because the model saw the frames as a sequence. Reading that
residual out with the vocabulary shows what it encodes: if it is motion, the words should be
actions ("climbing", "skiing", "driving") rather than things.

The image and video inputs of a model sit in slightly different places, so residuals are centred
over the sample before reading them out. SigLIP 2's video vector *is* its frame average, so its
residual is zero: the control.

    uv run python -m experiments.residual qwen3vl gemma2
"""

import sys

import numpy as np

from experiments.common import model, save, segments, space, table
from lens.readout import unit
from lens.videos import SOURCES
from lens.videos import segments as video_segments
from lens.vocab import kinetics400

PER_VIDEO = 8
TOP = 10


def sample():
    picked = []
    for source in SOURCES:
        segs = video_segments(source)
        step = max(len(segs) // PER_VIDEO, 1)
        picked += segs[step // 2 :: step][:PER_VIDEO]
    return picked


def is_action(phrase: str, actions: set[str]) -> bool:
    return phrase in actions or phrase.split()[0].endswith("ing")


def run(name: str) -> dict:
    keys, videos = segments(name)
    s = space(name, "medium")
    actions = {a.lower() for a in kinetics400()}
    picked = sample()
    index = {k: i for i, k in enumerate(keys)}
    video_vecs = videos[[index[seg.key] for seg in picked]]
    frame_means = np.stack([
        unit(model(name).embed_images(seg.frames).mean(axis=0)) for seg in picked
    ])
    residuals = video_vecs - frame_means
    residuals = unit(residuals - residuals.mean(axis=0))

    examples, plain_action, resid_action = [], [], []
    for seg, v, r in zip(picked, video_vecs, residuals):
        plain = [p for p, _ in s.top(v, TOP, "centered_csls")]
        along = s.vocab.centered @ r
        resid = [s.phrases[i] for i in np.argsort(-along)[:TOP]]
        plain_action.append(np.mean([is_action(p, actions) for p in plain]))
        resid_action.append(np.mean([is_action(p, actions) for p in resid]))
        examples.append({"segment": seg.key, "content": plain, "motion": resid})
    return {
        "model": name,
        "video_vs_frames_cos": float(np.mean(np.sum(video_vecs * frame_means, axis=1))),
        "action_share_content": float(np.mean(plain_action)),
        "action_share_residual": float(np.mean(resid_action)),
        "examples": examples,
    }


if __name__ == "__main__":
    results = [run(name) for name in sys.argv[1:]]
    save("residual", results)
    cols = ["model", "video_vs_frames_cos", "action_share_content", "action_share_residual"]
    print(table(results, cols))
    for r in results:
        print(f"\n{r['model']}")
        for e in r["examples"][::6]:
            print(f"  {e['segment']:>14}  content: {', '.join(e['content'][:5])}")
            print(f"  {'':>14}  motion:  {', '.join(e['motion'][:5])}")
