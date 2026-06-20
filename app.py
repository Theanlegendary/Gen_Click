import asyncio
import contextlib
import io
import json
import os
import re
import socket
import traceback
import uuid
from pathlib import Path

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from image_generator import COMMONS_API_URL, download_image, generate_image, make_fallback_queries
import shorts_generator


load_dotenv()

ROOT = Path(__file__).resolve().parent
VIDEO_REGISTRY = {}
ASSET_DIR = ROOT / "selected_assets"
ASSET_DIR.mkdir(exist_ok=True)
ASSET_REGISTRY = {}
ALLOWED_ASSET_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".mp4"}

app = FastAPI(title="Gen Click Shorts Studio")


class GenerateRequest(BaseModel):
    topic: str
    voice: str = "male"
    use_music: bool = False
    visual_count: int = 12
    subtitle_position: str = "middle"
    selected_visual_ids: list[str] = []


def has_valid_env(name):
    value = os.getenv(name)
    return bool(value and not value.startswith("your_") and value != "")


def environment_status():
    checks = [
        ("Gemini", "GEMINI_API_KEY", False),
        ("Pexels", "PEXELS_API_KEY", False),
        ("Pixabay", "PIXABAY_API_KEY", False),
    ]
    lines = []
    ready = True
    for label, env_name, required in checks:
        ok = has_valid_env(env_name)
        if required and not ok:
            ready = False
        if ok:
            mark = "OK"
        elif env_name == "GEMINI_API_KEY":
            mark = "Demo fallback"
        else:
            mark = "Missing" if required else "Optional"
        lines.append({"label": label, "status": mark, "required": required})
    return ready, lines


def infer_search_extra(topic):
    text = topic.lower()
    if any(term in text for term in ("black hole", "blackhole", "blackholes", "space", "planet", "mars", "moon", "star", "galaxy", "universe")):
        return "space astronomy cosmos universe"
    if any(term in text for term in ("ocean", "sea", "marine", "jellyfish", "deep")):
        return "ocean marine underwater nature"
    if any(term in text for term in ("ancient", "city", "ruin", "empire", "history")):
        return "ancient history archaeology civilization"
    if any(term in text for term in ("animal", "wildlife", "bird", "fish")):
        return "wildlife animal nature documentary"
    return "science discovery documentary"


def normalize_topic_for_search(topic):
    text = topic.lower()
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\b(facts?|things?|about|why|how|what|top|amazing|unknown|secret|secrets|mind|blowing|video|shorts?|scientists?|science|found|find|discovered?|tist)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    words = [word for word in text.split() if len(word) > 2]
    query = " ".join(words).strip()

    search_extra = infer_search_extra(topic)

    if "black hole" in query or "black holes" in query or "blackhole" in query or "blackholes" in query:
        query = "black hole space astronomy"
    elif query:
        query = f"{query} {search_extra}".strip()
    else:
        query = f"{topic} {search_extra}".strip()

    return " ".join(dict.fromkeys(query.split()))


def build_post_package(topic):
    clean_topic = " ".join(topic.split()).strip()
    title = f"{clean_topic.title()} Will Change How You See Earth"
    description = (
        f"Discover the hidden story behind {clean_topic}. "
        "This short from Discover World by Codex explores the surprising science, history, "
        "and mystery behind the universe around us.\n\n"
        "Watch until the end, then comment what hidden story we should discover next."
    )
    hashtags = [
        "#DiscoverWorldByCodex",
        "#WorldFacts",
        "#ScienceShorts",
        "#HistoryFacts",
        "#EarthMysteries",
        "#DidYouKnow",
        "#YouTubeShorts",
    ]
    return {
        "title": title[:95],
        "description": description,
        "hashtags": " ".join(hashtags),
    }


def fallback_viral_ideas():
    return [
        {
            "topic": "The ocean zone where sunlight disappears",
            "title": "The Ocean Turns Alien Below This Line",
            "description": "A 60-second discovery short about the depth where sunlight vanishes and life changes completely.",
            "hashtags": "#DiscoverWorldByCodex #OceanFacts #EarthMysteries #ScienceShorts #YouTubeShorts",
        },
        {
            "topic": "The ancient city hidden under the rainforest",
            "title": "A Lost City Was Hiding Under Trees",
            "description": "A discovery short about how modern scanning reveals forgotten worlds under dense forest.",
            "hashtags": "#DiscoverWorldByCodex #AncientHistory #LostCities #WorldFacts #YouTubeShorts",
        },
        {
            "topic": "Why deserts can suddenly bloom with flowers",
            "title": "The Desert Can Wake Up Overnight",
            "description": "A visual science short explaining the rare moments when dry land becomes a living carpet.",
            "hashtags": "#DiscoverWorldByCodex #NatureFacts #EarthScience #DesertBloom #YouTubeShorts",
        },
        {
            "topic": "The mountain that grows taller every year",
            "title": "This Mountain Is Still Rising",
            "description": "A fast discovery short about plate tectonics, pressure, and a planet that is still moving.",
            "hashtags": "#DiscoverWorldByCodex #Geology #MountainFacts #EarthMysteries #YouTubeShorts",
        },
    ]


def suggest_viral_idea(topic=""):
    topic = " ".join((topic or "").split()).strip()
    if has_valid_env("GEMINI_API_KEY"):
        try:
            import google.generativeai as genai

            model = genai.GenerativeModel(
                "gemini-2.5-flash",
                generation_config={"response_mime_type": "application/json"},
            )
            if topic:
                prompt = (
                    "Improve this user topic into one concise viral YouTube Shorts idea for Discover World by Codex. "
                    f"User topic: {topic}. "
                    "Return strict JSON with topic, title, description, and hashtags. "
                    "Keep topic under 12 words, title under 70 characters, description 1 sentence, hashtags 5-7 tags."
                )
            else:
                prompt = (
                    "Create one concise viral YouTube Shorts idea for a channel named Discover World by Codex. "
                    "Return strict JSON with topic, title, description, and hashtags. "
                    "Keep topic under 12 words, title under 70 characters, description 1 sentence, hashtags 5-7 tags."
                )
            data = json.loads(model.generate_content(prompt, request_options={"timeout": 15}).text)
            if all(key in data for key in ("topic", "title", "description", "hashtags")):
                return data
        except Exception:
            pass

    if topic:
        search_topic = normalize_topic_for_search(topic)
        display_topic = search_topic.title()
        if "Black Hole" in display_topic:
            display_topic = "Top 5 Black Holes Scientists Found"
        return {
            "topic": display_topic,
            "title": f"{display_topic} Will Bend Your Mind",
            "description": f"A short discovery video revealing the most surprising facts behind {display_topic}.",
            "hashtags": "#DiscoverWorldByCodex #ScienceShorts #DidYouKnow #YouTubeShorts #WorldFacts",
        }

    index = uuid.uuid4().int % len(fallback_viral_ideas())
    return fallback_viral_ideas()[index]


def register_asset(path):
    resolved = Path(path).resolve()
    if not resolved.exists() or ASSET_DIR not in resolved.parents:
        raise ValueError("Asset path is outside the asset directory.")
    asset_id = uuid.uuid4().hex
    ASSET_REGISTRY[asset_id] = str(resolved)
    return {
        "id": asset_id,
        "url": f"/asset/{asset_id}",
        "name": resolved.name,
        "kind": "video" if resolved.suffix.lower() == ".mp4" else "image",
    }


def safe_asset_name(filename):
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()
    cleaned = "".join(c if c.isalnum() else "_" for c in stem).strip("_")[:40] or "asset"
    return f"{int(uuid.uuid4().int % 1_000_000_000)}_{cleaned}{suffix}"


def fetch_topic_assets(topic, limit=6):
    topic = topic.strip()
    if not topic:
        raise ValueError("Enter a topic before fetching visuals.")

    headers = {"User-Agent": "Mozilla/5.0"}
    candidates = []
    for query in make_fallback_queries(topic):
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": str(max(limit * 3, 12)),
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "iiurlwidth": "720",
            "format": "json",
        }
        response = requests.get(COMMONS_API_URL, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        pages = (response.json().get("query") or {}).get("pages", {})
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            mime = (info.get("mime") or "").lower()
            url = info.get("thumburl") or info.get("url")
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            if url and mime in {"image/jpeg", "image/png"} and width >= 300 and height >= 300:
                candidates.append(url)
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break

    assets = []
    for index, url in enumerate(candidates[:limit], start=1):
        suffix = ".jpg" if ".jpg" in url.lower() or ".jpeg" in url.lower() else ".png"
        output_path = ASSET_DIR / f"fetched_{int(uuid.uuid4().int % 1_000_000_000)}_{index}{suffix}"
        try:
            download_image(url, str(output_path), max_retries=1)
            assets.append(register_asset(output_path))
        except Exception:
            continue

    if not assets:
        raise ValueError("No downloadable visuals found. Try a more specific topic or upload images.")
    return assets


def fetch_pixabay_assets(topic, limit=6):
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("PIXABAY_API_KEY is missing in .env.")

    search_query = normalize_topic_for_search(topic)
    search_queries = [search_query]
    if "black hole" in search_query:
        search_queries.extend(["black holes", "black hole universe", "space black hole"])

    hits = []
    seen_ids = set()
    for query in search_queries:
        params = {
            "key": api_key,
            "q": query,
            "image_type": "all",
            "safesearch": "true",
            "per_page": 50,
            "order": "popular",
        }
        response = requests.get("https://pixabay.com/api/", params=params, timeout=20)
        response.raise_for_status()
        for hit in response.json().get("hits", []):
            hit_id = hit.get("id")
            if hit_id in seen_ids:
                continue
            seen_ids.add(hit_id)
            hits.append(hit)

    relevant_terms = [term for term in search_query.split() if term not in {"space", "astronomy", "cosmos", "universe"}]

    def hit_score(hit):
        haystack = " ".join([
            str(hit.get("tags") or ""),
            str(hit.get("pageURL") or ""),
            str(hit.get("user") or ""),
        ]).lower()
        score = 0
        for term in relevant_terms:
            if term in haystack:
                score += 3
        if "black hole" in search_query and "black hole" in haystack:
            score += 10
        score += int(hit.get("likes") or 0) / 100
        score += int(hit.get("downloads") or 0) / 10000
        return score

    hits = sorted(hits, key=hit_score, reverse=True)
    assets = []
    for index, hit in enumerate(hits, start=1):
        if len(assets) >= limit:
            break
        if relevant_terms and hit_score(hit) <= 0:
            continue
        url = hit.get("largeImageURL") or hit.get("webformatURL") or hit.get("previewURL")
        if not url:
            continue
        output_path = ASSET_DIR / f"pixabay_{int(uuid.uuid4().int % 1_000_000_000)}_{index}.jpg"
        try:
            download_image(url, str(output_path), max_retries=1)
            assets.append(register_asset(output_path))
        except Exception:
            continue
    if not assets:
        raise ValueError(f"Pixabay did not return relevant images for '{search_query}'. Try AI images or upload your own visuals.")
    return assets


def generate_ai_assets(topic, limit=6):
    topic = topic.strip()
    if not topic:
        raise ValueError("Enter a topic before generating AI images.")
    search_query = normalize_topic_for_search(topic)
    prompt_templates = [
        "cinematic documentary opening scene about {query}, vertical 9:16, realistic, no text, no watermark",
        "close up scientific discovery visual about {query}, dramatic lighting, vertical 9:16, no text, no watermark",
        "wide immersive discovery scene showing {query}, high detail, vertical 9:16, no text",
        "educational visual explaining {query}, realistic documentary style, vertical 9:16, no text",
        "mysterious discovery scene connected to {query}, cinematic, vertical 9:16, no text",
        "epic final discovery shot about {query}, documentary style, vertical 9:16, no text",
    ]
    assets = []
    for index in range(1, limit + 1):
        template = prompt_templates[(index - 1) % len(prompt_templates)]
        prompt = template.format(topic=topic, query=search_query)
        if index > len(prompt_templates):
            prompt = f"{prompt}, unique variation {index}, different angle, different composition"
        output_path = ASSET_DIR / f"ai_{int(uuid.uuid4().int % 1_000_000_000)}_{index}.jpg"
        image_url = generate_image(prompt, index)
        download_image(image_url, str(output_path), max_retries=2)
        assets.append(register_asset(output_path))
    return assets


def find_free_port(start=7860, attempts=50):
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free local port found from {start} to {start + attempts - 1}.")


def run_generation(req):
    topic = req.topic.strip()
    if not topic:
        raise ValueError("Enter a topic first.")

    shorts_generator.USE_PEXELS_VIDEOS = False
    shorts_generator.FAST_IMAGE_DOWNLOAD_MODE = False
    requested_scene_count = max(3, min(int(req.visual_count), 15))
    selected_paths = []
    for asset_id in req.selected_visual_ids[:requested_scene_count]:
        path = ASSET_REGISTRY.get(asset_id)
        if path and Path(path).exists():
            selected_paths.append(path)
    previous_scene_count = shorts_generator.SCENE_COUNT
    shorts_generator.SCENE_COUNT = max(requested_scene_count, len(selected_paths), 3)
    shorts_generator.SELECTED_VISUAL_PATHS = selected_paths

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            output_path = asyncio.run(
                shorts_generator.create_shorts_video(
                    topic=topic,
                    voice_name=req.voice,
                    use_bg_music=req.use_music,
                    subtitle_position=req.subtitle_position,
                )
            )
    finally:
        shorts_generator.SELECTED_VISUAL_PATHS = []
        shorts_generator.SCENE_COUNT = previous_scene_count
    return output_path, buffer.getvalue(), build_post_package(topic)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/status")
def status():
    ready, checks = environment_status()
    return {"ready": ready, "checks": checks}


@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        output_path, log, post = await asyncio.to_thread(run_generation, req)
        resolved = Path(output_path).resolve()
        if not resolved.exists() or ROOT not in resolved.parents:
            raise ValueError("Generated video path is invalid.")
        token = uuid.uuid4().hex
        VIDEO_REGISTRY[token] = str(resolved)
        return JSONResponse({
            "ok": True,
            "message": f"Generated: {resolved}",
            "video_url": f"/video/{token}",
            "post": post,
            "log": log[-8000:],
        })
    except Exception:
        return JSONResponse({
            "ok": False,
            "message": "Generation failed. Check the log.",
            "log": traceback.format_exc()[-8000:],
        }, status_code=400)


@app.get("/suggest")
async def suggest(topic: str = ""):
    idea = await asyncio.to_thread(suggest_viral_idea, topic)
    return idea


@app.post("/upload-assets")
async def upload_assets(files: list[UploadFile] = File(...)):
    assets = []
    for upload in files[:15]:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in ALLOWED_ASSET_EXTENSIONS:
            continue
        filename = safe_asset_name(upload.filename or f"asset{suffix}")
        output_path = ASSET_DIR / filename
        content = await upload.read()
        output_path.write_bytes(content)
        assets.append(register_asset(output_path))
    if not assets:
        return JSONResponse({"ok": False, "message": "No valid image/video files uploaded."}, status_code=400)
    return {"ok": True, "assets": assets}


@app.get("/fetch-assets")
async def fetch_assets(topic: str, limit: int = 6):
    try:
        limit = min(max(int(limit), 3), 15)
        assets = await asyncio.to_thread(fetch_topic_assets, topic, limit)
        return {"ok": True, "assets": assets}
    except Exception as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)


@app.get("/fetch-pixabay-assets")
async def fetch_pixabay(topic: str, limit: int = 6):
    try:
        limit = min(max(int(limit), 3), 15)
        assets = await asyncio.to_thread(fetch_pixabay_assets, topic, limit)
        return {"ok": True, "assets": assets}
    except Exception as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)


@app.get("/generate-ai-assets")
async def generate_ai(topic: str, limit: int = 6):
    try:
        limit = min(max(int(limit), 3), 15)
        assets = await asyncio.to_thread(generate_ai_assets, topic, limit)
        return {"ok": True, "assets": assets}
    except Exception as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)


@app.get("/asset/{asset_id}")
def asset(asset_id: str):
    path = ASSET_REGISTRY.get(asset_id)
    if not path:
        raise HTTPException(status_code=404, detail="Asset not found.")
    suffix = Path(path).suffix.lower()
    media_type = "video/mp4" if suffix == ".mp4" else "image/jpeg"
    return FileResponse(path, media_type=media_type, filename=Path(path).name)


@app.get("/video/{token}")
def video(token: str):
    path = VIDEO_REGISTRY.get(token)
    if not path:
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(path, media_type="video/mp4", filename=Path(path).name)


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gen Click Shorts Studio</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #171717;
      --muted: #646464;
      --line: #ddddda;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --bad: #b42318;
      --ok: #027a48;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 28px auto;
      display: grid;
      grid-template-columns: 390px 1fr;
      gap: 18px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }
    p {
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.45;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 700;
      margin: 14px 0 6px;
    }
    textarea, select, input[type="file"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    textarea {
      min-height: 112px;
      resize: vertical;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 14px 0;
      color: var(--ink);
      font-weight: 600;
    }
    .check input { width: 18px; height: 18px; }
    button {
      width: 100%;
      min-height: 44px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button.secondary {
      background: #263238;
      margin-bottom: 10px;
    }
    button.secondary:hover { background: #111a1d; }
    button:disabled {
      cursor: wait;
      opacity: 0.68;
    }
    .status {
      display: grid;
      gap: 8px;
      margin-top: 14px;
    }
    .steps {
      margin: 0 0 14px 18px;
      padding: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }
    .pill {
      display: flex;
      justify-content: space-between;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .pill.ok strong { color: var(--ok); }
    .pill.bad strong { color: var(--bad); }
    .asset-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 10px;
    }
    .asset {
      position: relative;
      border: 2px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #111;
      aspect-ratio: 9 / 12;
      cursor: pointer;
    }
    .asset.selected { border-color: var(--accent); }
    .asset img, .asset video {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .asset span {
      position: absolute;
      top: 6px;
      right: 6px;
      min-width: 22px;
      height: 22px;
      border-radius: 999px;
      background: rgba(0,0,0,.72);
      color: #fff;
      display: grid;
      place-items: center;
      font-size: 12px;
      font-weight: 800;
    }
    video {
      width: 100%;
      max-height: 72vh;
      border-radius: 8px;
      background: #111;
    }
    .result {
      min-height: 44px;
      margin: 12px 0;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    pre {
      margin: 0;
      min-height: 260px;
      max-height: 440px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #111;
      color: #ededed;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>Gen Click Shorts Studio</h1>
      <p>Create a 60-second discovery short, preview the result, then use the post text for YouTube.</p>
      <ol class="steps">
        <li>Write a topic.</li>
        <li>Click Suggest Write if you want a cleaner idea.</li>
        <li>Pick visual count and subtitle position.</li>
        <li>Generate or fetch visuals, then select the good ones.</li>
        <li>Generate the final short.</li>
      </ol>

      <label for="topic">Topic</label>
      <textarea id="topic" placeholder="Example: why deserts can suddenly bloom with flowers"></textarea>
      <button id="suggest" class="secondary">Suggest Write</button>

      <div class="row">
        <div>
          <label for="voice">Voice</label>
          <select id="voice">
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="child">Child</option>
          </select>
        </div>
        <div>
          <label for="subtitlePosition">Subtitle</label>
          <select id="subtitlePosition">
            <option value="up">Up</option>
            <option value="middle" selected>Middle</option>
            <option value="bottom">Bottom</option>
          </select>
        </div>
      </div>

      <label class="check"><input id="music" type="checkbox"> Add background music</label>

      <label for="visualCount">Visual Count</label>
      <select id="visualCount">
        <option>3</option>
        <option>4</option>
        <option>5</option>
        <option>6</option>
        <option>7</option>
        <option>8</option>
        <option>9</option>
        <option>10</option>
        <option>11</option>
        <option selected>12</option>
        <option>13</option>
        <option>14</option>
        <option>15</option>
      </select>

      <label for="assetUpload">Selected Visuals</label>
      <input id="assetUpload" type="file" accept="image/*,video/mp4" multiple>
      <button id="uploadAssets" class="secondary">Upload Visuals</button>
      <button id="generateAiAssets" class="secondary">Generate AI Images</button>
      <button id="fetchPixabayAssets" class="secondary">Fetch From Pixabay</button>
      <button id="fetchAssets" class="secondary">Fetch From Wikimedia</button>
      <div id="assets" class="asset-grid"></div>

      <button id="generate">Generate Short</button>

      <div id="status" class="status"></div>
    </section>

    <section class="panel">
      <video id="video" controls></video>
      <div id="result" class="result">No video generated yet.</div>
      <label for="post">YouTube Post</label>
      <pre id="post">Title, description, and hashtags will appear here.</pre>
      <label for="log">Generation Log</label>
      <pre id="log">Waiting.</pre>
    </section>
  </main>

  <script>
    const button = document.getElementById("generate");
    const suggest = document.getElementById("suggest");
    const result = document.getElementById("result");
    const log = document.getElementById("log");
    const post = document.getElementById("post");
    const video = document.getElementById("video");
    const uploadAssets = document.getElementById("uploadAssets");
    const generateAiAssets = document.getElementById("generateAiAssets");
    const fetchPixabayAssets = document.getElementById("fetchPixabayAssets");
    const fetchAssets = document.getElementById("fetchAssets");
    const assetGrid = document.getElementById("assets");
    let assetList = [];
    let selectedAssetIds = [];

    function selectedVisualCount() {
      return Number(document.getElementById("visualCount").value || 12);
    }

    document.getElementById("visualCount").addEventListener("change", () => {
      selectedAssetIds = selectedAssetIds.slice(0, selectedVisualCount());
      renderAssets();
    });

    async function loadStatus() {
      const res = await fetch("/status");
      const data = await res.json();
      const status = document.getElementById("status");
      status.innerHTML = data.checks.map(item => {
        const cls = item.status === "OK" ? "ok" : (item.required ? "bad" : "");
        return `<div class="pill ${cls}"><span>${item.label}</span><strong>${item.status}</strong></div>`;
      }).join("");
    }

    function renderAssets() {
      assetGrid.innerHTML = assetList.map((asset) => {
        const selectedIndex = selectedAssetIds.indexOf(asset.id);
        const selected = selectedIndex >= 0;
        const media = asset.kind === "video"
          ? `<video src="${asset.url}" muted playsinline></video>`
          : `<img src="${asset.url}" alt="">`;
        const marker = selected ? selectedIndex + 1 : "";
        return `<div class="asset ${selected ? "selected" : ""}" data-id="${asset.id}">${media}<span>${marker}</span></div>`;
      }).join("");
    }

    assetGrid.addEventListener("click", (event) => {
      const tile = event.target.closest(".asset");
      if (!tile) return;
      const id = tile.dataset.id;
      if (selectedAssetIds.includes(id)) {
        selectedAssetIds = selectedAssetIds.filter(item => item !== id);
      } else if (selectedAssetIds.length < selectedVisualCount()) {
        selectedAssetIds.push(id);
      }
      renderAssets();
    });

    uploadAssets.addEventListener("click", async () => {
      const files = document.getElementById("assetUpload").files;
      if (!files.length) {
        result.textContent = "Choose image or MP4 files first.";
        return;
      }
      uploadAssets.disabled = true;
      uploadAssets.textContent = "Uploading...";
      const form = new FormData();
      Array.from(files).slice(0, selectedVisualCount()).forEach(file => form.append("files", file));
      try {
        const res = await fetch("/upload-assets", {method: "POST", body: form});
        const data = await res.json();
        if (!data.ok) throw new Error(data.message || "Upload failed.");
        assetList = data.assets;
        selectedAssetIds = assetList.map(asset => asset.id).slice(0, selectedVisualCount());
        renderAssets();
        result.textContent = `Loaded ${assetList.length} uploaded visuals.`;
      } catch (err) {
        result.textContent = String(err);
      } finally {
        uploadAssets.disabled = false;
        uploadAssets.textContent = "Upload Visuals";
      }
    });

    async function loadAssetsFromEndpoint(buttonEl, url, loadingText, doneText) {
      const topic = document.getElementById("topic").value.trim();
      if (!topic) {
        result.textContent = "Enter a topic first.";
        return;
      }
      buttonEl.disabled = true;
      buttonEl.textContent = loadingText;
      try {
        const res = await fetch(url + `?limit=${selectedVisualCount()}&topic=${encodeURIComponent(topic)}`);
        const data = await res.json();
        if (!data.ok) throw new Error(data.message || "Visual fetch failed.");
        assetList = data.assets;
        selectedAssetIds = assetList.map(asset => asset.id).slice(0, selectedVisualCount());
        renderAssets();
        result.textContent = `${doneText}: ${assetList.length} visuals loaded.`;
      } catch (err) {
        result.textContent = String(err);
      } finally {
        buttonEl.disabled = false;
      }
    }

    generateAiAssets.addEventListener("click", async () => {
      await loadAssetsFromEndpoint(generateAiAssets, "/generate-ai-assets", "Generating AI images...", "AI images ready");
      generateAiAssets.textContent = "Generate AI Images";
    });

    fetchPixabayAssets.addEventListener("click", async () => {
      await loadAssetsFromEndpoint(fetchPixabayAssets, "/fetch-pixabay-assets", "Fetching Pixabay...", "Pixabay visuals ready");
      fetchPixabayAssets.textContent = "Fetch From Pixabay";
    });

    fetchAssets.addEventListener("click", async () => {
      await loadAssetsFromEndpoint(fetchAssets, "/fetch-assets", "Fetching Wikimedia...", "Wikimedia visuals ready");
      fetchAssets.textContent = "Fetch From Wikimedia";
    });

    suggest.addEventListener("click", async () => {
      suggest.disabled = true;
      suggest.textContent = "Writing suggest...";
      try {
        const topic = document.getElementById("topic").value.trim();
        const res = await fetch(`/suggest?topic=${encodeURIComponent(topic)}`);
        const data = await res.json();
        document.getElementById("topic").value = data.topic || "";
        post.textContent = `Title: ${data.title || ""}\n\nDescription:\n${data.description || ""}\n\nHashtags:\n${data.hashtags || ""}`;
      } catch (err) {
        post.textContent = String(err);
      } finally {
        suggest.disabled = false;
        suggest.textContent = "Suggest Write";
      }
    });

    button.addEventListener("click", async () => {
      button.disabled = true;
      button.textContent = "Generating...";
      result.textContent = "Rendering. This can take a few minutes.";
      log.textContent = "Starting generation...";
      post.textContent = "Waiting for post text...";
      video.removeAttribute("src");

      try {
        const res = await fetch("/generate", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            topic: document.getElementById("topic").value,
            voice: document.getElementById("voice").value,
            use_music: document.getElementById("music").checked,
            visual_count: selectedVisualCount(),
            subtitle_position: document.getElementById("subtitlePosition").value,
            selected_visual_ids: selectedAssetIds
          })
        });
        const data = await res.json();
        result.textContent = data.message;
        log.textContent = data.log || "";
        if (data.post) {
          post.textContent = `Title: ${data.post.title}\n\nDescription:\n${data.post.description}\n\nHashtags:\n${data.post.hashtags}`;
        }
        if (data.ok && data.video_url) {
          video.src = data.video_url;
          video.load();
        }
      } catch (err) {
        result.textContent = "Request failed.";
        log.textContent = String(err);
      } finally {
        button.disabled = false;
        button.textContent = "Generate Short";
      }
    });

    loadStatus();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    configured_port = os.getenv("GEN_CLICK_PORT")
    server_port = int(configured_port) if configured_port else find_free_port()
    uvicorn.run(app, host="127.0.0.1", port=server_port)
