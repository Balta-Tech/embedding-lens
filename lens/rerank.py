"""Second stage: rerank a read-out's candidates with Qwen3-VL-Reranker-2B.

The embedding read-out compares one video vector with one phrase vector; the two never meet
inside a model. A reranker reads the video and a phrase together and answers "does this
describe it?" (the logit of "yes" minus "no"). That costs a full model pass per pair, so it is
only run on the embedding's top candidates.

Every pair for a moment starts with the same tokens (instruction, then the video), so the video
is encoded once and its key/value cache is reused for each candidate phrase.
"""

from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, DynamicCache, Qwen3VLForConditionalGeneration

from lens.models.base import device
from lens.videos import Segment

MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"
SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct '
    'provided. Note that the answer can only be "yes" or "no".'
)
INSTRUCTION = "Given a video, retrieve words or short phrases that describe what is visible in it."
# The reference script's video budget: at most 4 frame pairs' worth of 1280 visual tokens
TOTAL_PIXELS = 4 * 2 * 1280 * 32 * 32


class Reranker:
    def __init__(self, instruction: str = INSTRUCTION, batch_size: int = 8) -> None:
        path = Path(snapshot_download(MODEL_ID, local_files_only=True))
        lm = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16)
        lm.to(device()).eval()
        self.model = lm.model
        self.processor = AutoProcessor.from_pretrained(path, padding_side="right")
        vocab = self.processor.tokenizer.get_vocab()
        # Score = logit("yes") - logit("no") at the last position
        self.direction = (lm.lm_head.weight[vocab["yes"]] - lm.lm_head.weight[vocab["no"]]).detach()
        self.instruction = instruction
        self.batch_size = batch_size

    def _split_prompt(self, frames: list[str]):
        """Tokenised prefix (instruction and video) and the text that follows each phrase."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "text", "text": "<Instruct>: " + self.instruction},
                {"type": "text", "text": "<Query>:"},
                {"type": "video", "video": ["file://" + f for f in frames], "total_pixels": TOTAL_PIXELS},
                {"type": "text", "text": "\n<Document>:"},
                {"type": "text", "text": "{phrase}"},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix, suffix = text.split("{phrase}")
        _, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
        )
        videos, metadata = zip(*videos)
        inputs = self.processor(
            text=[prefix], videos=list(videos), video_metadata=list(metadata),
            do_resize=False, return_tensors="pt", **video_kwargs,
        )
        return inputs.to(self.model.device), suffix

    @torch.no_grad()
    def score(self, segment: Segment, phrases: list[str]) -> np.ndarray:
        """Reranker logit for each phrase as a description of the segment (higher is better)."""
        inputs, suffix = self._split_prompt([str(p) for p in segment.frames])
        self.model.rope_deltas = None
        prefix = self.model(**inputs, use_cache=True)
        layers = [(layer.keys, layer.values) for layer in prefix.past_key_values.layers]
        tokenizer = self.processor.tokenizer
        scores = []
        for start in range(0, len(phrases), self.batch_size):
            batch = phrases[start : start + self.batch_size]
            tokens = tokenizer([p + suffix for p in batch], padding=True, return_tensors="pt", add_special_tokens=False)
            ids = tokens["input_ids"].to(self.model.device)
            cache = DynamicCache()
            for i, (k, v) in enumerate(layers):
                cache.update(k.expand(len(batch), -1, -1, -1).contiguous(), v.expand(len(batch), -1, -1, -1).contiguous(), i)
            # No attention mask: positions continue from the prefix, and right padding sits after
            # every real token, so the causal mask already hides it
            hidden = self.model(input_ids=ids, past_key_values=cache, use_cache=True).last_hidden_state
            last = tokens["attention_mask"].sum(1).to(self.model.device) - 1
            final = hidden[torch.arange(len(batch), device=hidden.device), last]
            scores.append((final @ self.direction).float().cpu().numpy())
            del cache, hidden
        del prefix, layers
        # MPS keeps freed blocks cached; without this a long run creeps into swap
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return np.concatenate(scores)


def export(name: str, tier: str, k: int = 50, top: int = 14) -> None:
    """Rerank every segment's top k for the site: web/public/data/readouts/{model}.{tier}.rerank.json.

    Separate from lens.export because it takes a few seconds per segment.

        uv run python -m lens.rerank qwen3vl:medium siglip2:large
    """
    import json
    import time

    from lens.config import BEST_METHOD, ROOT
    from lens.index import load_segments, load_vocab
    from lens.readout import Space
    from lens.videos import all_segments

    keys, videos = load_segments(name)
    phrases, text = load_vocab(name, tier)
    space = Space(phrases, text, videos)
    by_key = {s.key: s for s in all_segments()}
    reranker = _shared()
    out = {}
    start = time.time()
    for i, (key, v) in enumerate(zip(keys, videos)):
        words = [p for p, _ in space.top(v, k, BEST_METHOD[name])]
        scores = reranker.score(by_key[key], words)
        order = np.argsort(-scores)[:top]
        # [phrase, reranker logit, rank in the embedding read-out]
        out[key] = [[words[j], round(float(scores[j]), 3), int(j) + 1] for j in order]
        if i % 25 == 0:
            print(f"{name}.{tier}: {i}/{len(keys)} ({time.time() - start:.0f}s)", flush=True)
    path = ROOT / "web" / "public" / "data" / "readouts" / f"{name}.{tier}.rerank.json"
    path.write_text(json.dumps({"k": k, "segments": out}, separators=(",", ":")))


_reranker = None


def _shared() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


if __name__ == "__main__":
    import sys

    for target in sys.argv[1:]:
        export(*target.split(":"))
