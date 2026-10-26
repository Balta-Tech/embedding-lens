"""Download the example videos, extract frames and cut them into segments.

    uv run python -m lens.videos download   # fetch the sources at 720p
    uv run python -m lens.videos frames     # extract frames at FRAME_FPS
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from lens.config import FRAME_FPS, FRAME_WIDTH, FRAMES_DIR, SEGMENT_SECONDS, VIDEOS_DIR


@dataclass(frozen=True)
class Source:
    id: str
    youtube_id: str
    title: str
    # "real" or "sim": used by the simulation-vs-reality experiment
    world: str
    # Optional "start-end" section for long videos, in yt-dlp's *MM:SS-MM:SS syntax
    section: str | None = None


SOURCES = [
    Source("zugspitze", "8zeq6wXqP8k", "Zugspitze via Höllental, icy conditions", "real"),
    Source("wetterstein", "MA2pb7oi4PE", "Obere Wettersteinspitze, no commentary", "real"),
    Source("kart", "TuOJHMUfayY", "Battaring Silver kart race", "real", section="05:00-10:00"),
    Source("f16", "gVSsC1JE_ho", "F-16 cold start, take-off and landing in DCS", "sim"),
    Source("assetto", "eDUi-Tqt090", "Assetto Corsa, early cut", "sim"),
    Source("trex", "r4iIBc5ZeF8", "A bot playing the Chrome T-Rex game", "sim"),
]
SOURCES_BY_ID = {s.id: s for s in SOURCES}


@dataclass(frozen=True)
class Segment:
    video_id: str
    index: int
    start: float
    end: float
    frames: tuple[Path, ...]

    @property
    def key(self) -> str:
        return f"{self.video_id}:{self.index}"


def video_path(source: Source) -> Path:
    return VIDEOS_DIR / f"{source.id}.mp4"


def download(source: Source) -> Path:
    out = video_path(source)
    if out.exists():
        return out
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--js-runtimes", "node",
        # H.264 + AAC so every browser can play the file directly
        "-f", "bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/b[height<=720][ext=mp4]",
        "--merge-output-format", "mp4",
        "-o", str(out),
        f"https://www.youtube.com/watch?v={source.youtube_id}",
    ]
    if source.section:
        cmd[3:3] = ["--download-sections", f"*{source.section}", "--force-keyframes-at-cuts"]
    subprocess.run(cmd, check=True)
    return out


def duration_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def frames_dir(source: Source) -> Path:
    return FRAMES_DIR / source.id


def extract_frames(source: Source) -> list[Path]:
    """Sample frames at FRAME_FPS. Frame n (1-based) is taken at (n - 0.5) / FRAME_FPS seconds."""
    out_dir = frames_dir(source)
    existing = sorted(out_dir.glob("*.jpg"))
    if existing:
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(video_path(source)),
            "-vf", f"fps={FRAME_FPS},scale={FRAME_WIDTH}:-2",
            "-q:v", "3", str(out_dir / "%05d.jpg"),
        ],
        check=True,
    )
    return sorted(out_dir.glob("*.jpg"))


def segments(source: Source, seconds: float = SEGMENT_SECONDS) -> list[Segment]:
    frames = sorted(frames_dir(source).glob("*.jpg"))
    per_segment = int(seconds * FRAME_FPS)
    result = []
    for index, first in enumerate(range(0, len(frames), per_segment)):
        chunk = tuple(frames[first : first + per_segment])
        # Drop a trailing stub shorter than half a segment: too little to say anything about
        if len(chunk) < per_segment / 2:
            break
        start = first / FRAME_FPS
        result.append(Segment(source.id, index, start, start + len(chunk) / FRAME_FPS, chunk))
    return result


def all_segments() -> list[Segment]:
    return [seg for source in SOURCES for seg in segments(source)]


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    for source in SOURCES:
        if command in ("download", "all"):
            path = download(source)
            print(f"{source.id}: {duration_seconds(path):.0f}s  {path.name}")
        if command in ("frames", "all"):
            frames = extract_frames(source)
            print(f"{source.id}: {len(frames)} frames, {len(segments(source))} segments")


if __name__ == "__main__":
    main()
