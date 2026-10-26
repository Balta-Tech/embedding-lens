"""Vocabularies of words and phrases to read embeddings out with, in three sizes.

    small   ~4.6k  RAM++ tag list: common visual tags (objects, scenes, attributes, actions)
    medium  ~23k   + ImageNet-21k classes, Places365 scenes, Kinetics-400 actions
    large   ~119k  + every WordNet noun lemma

Each tier contains the one below it. Build with:

    uv run python -m lens.vocab
"""

import re
import urllib.request

from lens.config import VOCAB_DIR

TIERS = ("small", "medium", "large")

SOURCES = {
    "ram": "https://raw.githubusercontent.com/xinyu1205/recognize-anything/main/ram/data/ram_tag_list.txt",
    "imagenet21k": "https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/data/_info/imagenet22k_synsets.txt",
    "places365": "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt",
    "kinetics400": "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt",
}


def _fetch(name: str) -> list[str]:
    cache = VOCAB_DIR / "sources" / f"{name}.txt"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCES[name], timeout=60) as response:
            cache.write_bytes(response.read())
    return cache.read_text(encoding="utf-8").splitlines()


def clean(phrase: str) -> str:
    phrase = phrase.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", phrase)


def ram() -> list[str]:
    return _fetch("ram")


def _wordnet():
    import nltk
    from nltk.corpus import wordnet

    nltk.download("wordnet", download_dir=str(VOCAB_DIR / "nltk"), quiet=True)
    nltk.data.path.insert(0, str(VOCAB_DIR / "nltk"))
    return wordnet


def imagenet21k() -> list[str]:
    # The list holds WordNet synset IDs like "n01440764"; keep each synset's first, most common name
    wordnet = _wordnet()
    return [wordnet.synset_from_pos_and_offset("n", int(sid[1:])).lemma_names()[0]
            for sid in _fetch("imagenet21k")]


def places365() -> list[str]:
    # "/a/airfield 0" or "/b/bedchamber 37" -> "airfield"; "/f/field/wild 3" -> "wild field"
    names = []
    for line in _fetch("places365"):
        parts = line.split()[0].strip("/").split("/")[1:]
        names.append(" ".join(reversed(parts)))
    return names


def kinetics400() -> list[str]:
    return _fetch("kinetics400")


def wordnet_nouns() -> list[str]:
    wordnet = _wordnet()
    return [lemma for synset in wordnet.all_synsets("n") for lemma in synset.lemma_names()]


def _unique(*lists: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for phrases in lists:
        for p in phrases:
            p = clean(p)
            # Skip empties, single characters and bare numbers
            if len(p) > 1 and not p.isdigit():
                seen.setdefault(p, None)
    return list(seen)


def build() -> dict[str, list[str]]:
    small = _unique(ram())
    medium = _unique(small, imagenet21k(), places365(), kinetics400())
    large = _unique(medium, wordnet_nouns())
    tiers = {"small": small, "medium": medium, "large": large}
    for name, phrases in tiers.items():
        (VOCAB_DIR / f"{name}.txt").write_text("\n".join(phrases) + "\n", encoding="utf-8")
    return tiers


def load(tier: str) -> list[str]:
    path = VOCAB_DIR / f"{tier}.txt"
    if not path.exists():
        build()
    return path.read_text(encoding="utf-8").splitlines()


if __name__ == "__main__":
    for name, phrases in build().items():
        print(f"{name:>7}: {len(phrases):>7,} phrases  e.g. {', '.join(phrases[1000:1005])}")
