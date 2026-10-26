"""Paths and defaults shared by the pipeline, the experiments and the API."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VIDEOS_DIR = DATA / "videos"
FRAMES_DIR = DATA / "frames"
VOCAB_DIR = DATA / "vocab"
EMBEDDINGS_DIR = DATA / "embeddings"
RESULTS_DIR = DATA / "results"

# Frames are sampled at this rate and resized to this width. 2 fps keeps short events
# (a turn, a jump) inside a segment; 640 px is above what every model resizes to anyway.
FRAME_FPS = 2
FRAME_WIDTH = 640

# Videos are cut into fixed windows of this many seconds. Short enough that a segment shows
# one thing, long enough to contain motion.
SEGMENT_SECONDS = 5
