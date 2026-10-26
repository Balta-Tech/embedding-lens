# embedding-lens

What does a multimodal embedding model actually *see* in a video?

embedding-lens cuts videos into segments, embeds each segment with a video-text embedding
model, and then reads the vector back out as words: the phrases from a large vocabulary that sit
closest to it in the shared text/video space. A local web app shows the result as a bubble chart,
a concept timeline, and a game where you describe a moment and compare your words with the model's.

Work in progress.
