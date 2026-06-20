# Gen Click Shorts Studio

Local AI video generator for vertical YouTube Shorts.

## Setup

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Create `.env` from `.env.example` and add at least:

```bash
GEMINI_API_KEY=your_real_key_here
```

`PEXELS_API_KEY` is optional. Add it if you want stock video search.

3. Start the web app:

```bash
python3 app.py
```

Open `http://127.0.0.1:7860`.

## Modes

- `AI images first`: uses generated images, then falls back to web images and gradients.
- `Stock video first`: tries Pexels video before generated images.
- `Fast web images`: uses downloadable web images for faster tests.

Generated videos are saved in timestamped `short_*` folders.
