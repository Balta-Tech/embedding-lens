<p align="center">
  <img src="docs/banner.svg" alt="embedding-lens: a video segment and the words closest to its embedding" width="100%">
</p>

<p align="center">
  <a href="https://embedding-lens.balta-tech.com"><b>Live site</b></a> ·
  <a href="https://balta-tech.com">Balta Tech</a>
</p>

---

**embedding-lens** reads a video embedding back out as words. Each 5-second segment of a video
is embedded into a space it shares with text; the phrases from a large vocabulary that land
closest to it show what the model picked up: *mountain ridge*, *snow*, *helmet camera*.

## How it works

```
video ─▶ frames (2 fps) ─▶ 5 s segments ─▶ segment vector ┐
                                                          ├─▶ cosine ─▶ calibrated scores ─▶ words
vocabulary (4.5k / 21k / 100k+ phrases) ─▶ text vectors ──┘
```

Raw cosine similarity is not used directly: scores are calibrated against a neutral baseline and
corrected for "hub" phrases that sit close to everything. The [experiments](experiments/) show
which corrections help and by how much.

## Models

| Model | Input | Role |
| --- | --- | --- |
| [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | native video | main model |
| [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | native video + audio | comparison |
| [SigLIP 2 base](https://huggingface.co/google/siglip2-base-patch16-256) | single frames, averaged | baseline |

All run locally on Apple silicon (tested on an M3 Pro, 18 GB).

## Run it locally

Requirements: [uv](https://docs.astral.sh/uv/), Node 22+, ffmpeg.

```bash
uv sync
uv run python -m lens.videos all        # download the example videos, extract frames
npm install && npm run dev              # web app on http://localhost:5173
```

## Deploy

The site is a static build served by the `embedding-lens` Cloudflare Worker. Every push to `main`
runs [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml): `npm run build`, then
`wrangler deploy`. The workflow needs one repository secret, `CLOUDFLARE_API_TOKEN`.

The embedding backend is not deployed: the site ships with precomputed results and uses the local
backend for live queries when it is reachable.

## Example videos

From [youtube.com/@rbalazs](https://www.youtube.com/@rbalazs). They are downloaded locally and
never committed.
