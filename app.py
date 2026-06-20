import asyncio
import contextlib
import io
import json
import os
import re
import socket
import sys
import threading
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
    manual_script: str = ""
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
    original_text = topic.lower()
    text = topic.lower()
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\b(facts?|things?|about|why|how|what|top|amazing|unknown|secret|secrets|mind|blowing|video|shorts?|scientists?|science|found|find|discovered?|tist)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    words = [word for word in text.split() if len(word) > 2]
    query = " ".join(words).strip()

    search_extra = infer_search_extra(topic)

    if "black hole" in query or "black holes" in query or "blackhole" in query or "blackholes" in query:
        query = "black hole space astronomy"
    elif "nikola tesla" in original_text or "niko tesla" in original_text:
        query = "nikola tesla"
    elif len(words) == 2 and all(word[:1].isalpha() for word in words):
        query = " ".join(words)
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


def fetch_topic_assets(topic, limit=6, start_offset=0, progress_callback=None):
    topic = topic.strip()
    if not topic:
        raise ValueError("Enter a topic before fetching visuals.")

    if progress_callback:
        progress_callback(10, "Searching Wikimedia Commons...")

    headers = {"User-Agent": "Mozilla/5.0"}
    candidates = []
    for query in make_fallback_queries(topic):
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "iiurlwidth": "720",
            "format": "json",
        }
        try:
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
                    if url not in candidates:
                        candidates.append(url)
        except Exception:
            continue

    if progress_callback:
        progress_callback(30, f"Found {len(candidates)} candidates. Downloading requested batch...")

    assets = []
    slice_candidates = candidates[start_offset : start_offset + limit]
    if not slice_candidates:
        if start_offset > 0:
            raise ValueError(f"No more visuals found on Wikimedia for '{topic}'. Try a different source.")
        else:
            raise ValueError("No downloadable visuals found. Try a more specific topic or upload images.")

    total_to_download = len(slice_candidates)
    for index, url in enumerate(slice_candidates, start=1):
        suffix = ".jpg" if ".jpg" in url.lower() or ".jpeg" in url.lower() else ".png"
        output_path = ASSET_DIR / f"fetched_{int(uuid.uuid4().int % 1_000_000_000)}_{index}{suffix}"
        try:
            download_image(url, str(output_path), max_retries=1)
            assets.append(register_asset(output_path))
        except Exception:
            pass
        if progress_callback:
            pct = 30 + int((index / total_to_download) * 70)
            progress_callback(pct, f"Downloaded {index}/{total_to_download} visuals...")

    if not assets:
        raise ValueError("No downloadable visuals found. Try a more specific topic or upload images.")
    return assets


def fetch_pixabay_assets(topic, limit=6, start_offset=0, progress_callback=None):
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("PIXABAY_API_KEY is missing in .env.")

    if progress_callback:
        progress_callback(10, "Searching Pixabay...")

    search_query = normalize_topic_for_search(topic)
    exact_topic = " ".join(re.sub(r"[^a-zA-Z0-9\s-]", " ", topic).split()).lower()
    search_queries = [search_query]
    if exact_topic and exact_topic not in search_queries:
        search_queries.insert(0, exact_topic)
    if "black hole" in search_query:
        search_queries.extend(["black holes", "black hole universe", "space black hole"])
    if "nikola tesla" in search_query:
        search_queries.extend(["nikola tesla", "tesla inventor"])

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
        try:
            response = requests.get("https://pixabay.com/api/", params=params, timeout=20)
            response.raise_for_status()
            for hit in response.json().get("hits", []):
                hit_id = hit.get("id")
                if hit_id in seen_ids:
                    continue
                seen_ids.add(hit_id)
                hits.append(hit)
        except Exception:
            continue

    relevant_terms = [term for term in search_query.split() if term not in {"space", "astronomy", "cosmos", "universe", "science", "discovery", "documentary"}]
    exact_phrase = " ".join(relevant_terms)

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
        if exact_phrase and exact_phrase in haystack:
            score += 12
        if "black hole" in search_query and "black hole" in haystack:
            score += 10
        if "nikola tesla" in search_query and "nikola tesla" in haystack:
            score += 15
        score += int(hit.get("likes") or 0) / 100
        score += int(hit.get("downloads") or 0) / 10000
        return score

    hits = sorted(hits, key=hit_score, reverse=True)
    if relevant_terms:
        hits = [hit for hit in hits if hit_score(hit) > 0]

    if progress_callback:
        progress_callback(30, f"Found {len(hits)} matching visuals. Downloading batch...")

    slice_hits = hits[start_offset : start_offset + limit]
    if not slice_hits:
        if start_offset > 0:
            raise ValueError(f"No more visuals found on Pixabay for '{topic}'. Try a different source.")
        else:
            raise ValueError(f"Pixabay did not return relevant images for '{search_query}'. Try AI images or upload your own visuals.")

    assets = []
    total_to_download = len(slice_hits)
    for index, hit in enumerate(slice_hits, start=1):
        url = hit.get("largeImageURL") or hit.get("webformatURL") or hit.get("previewURL")
        if not url:
            continue
        output_path = ASSET_DIR / f"pixabay_{int(uuid.uuid4().int % 1_000_000_000)}_{index}.jpg"
        try:
            download_image(url, str(output_path), max_retries=1)
            assets.append(register_asset(output_path))
        except Exception:
            pass
        if progress_callback:
            pct = 30 + int((index / total_to_download) * 70)
            progress_callback(pct, f"Downloaded {index}/{total_to_download} visuals...")

    if not assets:
        raise ValueError(f"Pixabay did not return relevant images for '{search_query}'. Try AI images or upload your own visuals.")
    return assets


def generate_ai_assets(topic, limit=6, start_offset=0, progress_callback=None):
    topic = topic.strip()
    if not topic:
        raise ValueError("Enter a topic before generating AI images.")

    if progress_callback:
        progress_callback(10, "Preparing prompts for AI generation...")

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
    
    for index in range(start_offset + 1, start_offset + limit + 1):
        template = prompt_templates[(index - 1) % len(prompt_templates)]
        prompt = template.format(topic=topic, query=search_query)
        if index > len(prompt_templates):
            prompt = f"{prompt}, unique variation {index}, different angle, different composition"
        output_path = ASSET_DIR / f"ai_{int(uuid.uuid4().int % 1_000_000_000)}_{index}.jpg"
        
        if progress_callback:
            progress_callback(10 + int(((index - start_offset - 1) / limit) * 90), f"Generating AI image {index - start_offset}/{limit}...")
            
        try:
            image_url = generate_image(prompt, index)
            download_image(image_url, str(output_path), max_retries=2)
            assets.append(register_asset(output_path))
        except Exception as e:
            print(f"Error generating AI image: {e}")
            continue

    if progress_callback:
        progress_callback(100, "AI generation complete!")
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


TASKS = {}

class TaskLogStream(io.IOBase):
    def __init__(self, task_id, original_stdout):
        self.task_id = task_id
        self.original_stdout = original_stdout
    def write(self, s):
        if self.task_id in TASKS:
            TASKS[self.task_id]["log"] += s
        self.original_stdout.write(s)
        self.original_stdout.flush()
        return len(s)

def update_task_progress(task_id, percent, message, status="running", result=None):
    if task_id in TASKS:
        TASKS[task_id]["percent"] = percent
        TASKS[task_id]["message"] = message
        TASKS[task_id]["status"] = status
        if result is not None:
            TASKS[task_id]["result"] = result

def run_fetch_assets_task(task_id, fetch_func, topic, limit, start_offset):
    def progress_cb(percent, message):
        update_task_progress(task_id, percent, message)
    try:
        assets = fetch_func(topic, limit, start_offset, progress_callback=progress_cb)
        update_task_progress(task_id, 100, "Done", status="completed", result={"ok": True, "assets": assets})
    except Exception as exc:
        traceback.print_exc()
        update_task_progress(task_id, 0, str(exc), status="failed")

def run_generation_task(task_id, req):
    try:
        output_path, post = run_generation(req, task_id)
        resolved = Path(output_path).resolve()
        if not resolved.exists() or ROOT not in resolved.parents:
            raise ValueError("Generated video path is invalid.")
        token = uuid.uuid4().hex
        VIDEO_REGISTRY[token] = str(resolved)
        
        result = {
            "ok": True,
            "message": f"Generated: {resolved}",
            "video_url": f"/video/{token}",
            "post": post,
            "log": TASKS[task_id]["log"][-8000:]
        }
        update_task_progress(task_id, 100, "Done", status="completed", result=result)
    except Exception as exc:
        traceback.print_exc()
        update_task_progress(task_id, 0, str(exc), status="failed")


def run_generation(req, task_id):
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

    def progress_cb(percent, message):
        update_task_progress(task_id, percent, message)

    stream = TaskLogStream(task_id, sys.stdout)
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            output_path = asyncio.run(
                shorts_generator.create_shorts_video(
                    topic=topic,
                    manual_script=req.manual_script,
                    voice_name=req.voice,
                    use_bg_music=req.use_music,
                    subtitle_position=req.subtitle_position,
                    progress_callback=progress_cb
                )
            )
    finally:
        shorts_generator.SELECTED_VISUAL_PATHS = []
        shorts_generator.SCENE_COUNT = previous_scene_count
    return output_path, build_post_package(topic)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/status")
def status():
    ready, checks = environment_status()
    return {"ready": ready, "checks": checks}


@app.get("/task/{task_id}")
def get_task_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.post("/generate")
async def generate(req: GenerateRequest):
    task_id = uuid.uuid4().hex
    TASKS[task_id] = {
        "status": "pending",
        "percent": 0,
        "message": "Starting video generation...",
        "log": "",
        "result": None
    }
    threading.Thread(target=run_generation_task, args=(task_id, req), daemon=True).start()
    return {"ok": True, "task_id": task_id}


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
async def fetch_assets(topic: str, limit: int = 6, start_offset: int = 0):
    limit = min(max(int(limit), 3), 15)
    task_id = uuid.uuid4().hex
    TASKS[task_id] = {
        "status": "pending",
        "percent": 0,
        "message": "Initializing...",
        "log": "",
        "result": None
    }
    threading.Thread(target=run_fetch_assets_task, args=(task_id, fetch_topic_assets, topic, limit, start_offset), daemon=True).start()
    return {"ok": True, "task_id": task_id}


@app.get("/fetch-pixabay-assets")
async def fetch_pixabay(topic: str, limit: int = 6, start_offset: int = 0):
    limit = min(max(int(limit), 3), 15)
    task_id = uuid.uuid4().hex
    TASKS[task_id] = {
        "status": "pending",
        "percent": 0,
        "message": "Initializing...",
        "log": "",
        "result": None
    }
    threading.Thread(target=run_fetch_assets_task, args=(task_id, fetch_pixabay_assets, topic, limit, start_offset), daemon=True).start()
    return {"ok": True, "task_id": task_id}


@app.get("/generate-ai-assets")
async def generate_ai(topic: str, limit: int = 6, start_offset: int = 0):
    limit = min(max(int(limit), 3), 15)
    task_id = uuid.uuid4().hex
    TASKS[task_id] = {
        "status": "pending",
        "percent": 0,
        "message": "Initializing...",
        "log": "",
        "result": None
    }
    threading.Thread(target=run_fetch_assets_task, args=(task_id, generate_ai_assets, topic, limit, start_offset), daemon=True).start()
    return {"ok": True, "task_id": task_id}


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

      <label for="manualScript">Manual Voiceover Script (optional)</label>
      <textarea id="manualScript" placeholder="Write your own narration here. If filled, the tool reads this instead of writing a script."></textarea>

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
      <button id="fetchMore" class="secondary" style="display: none; background: #0f766e; font-weight: bold; margin-bottom: 10px;">Fetch More</button>

      <div id="taskProgressContainer" style="display: none; margin: 14px 0; border: 1px solid var(--line); border-radius: 6px; padding: 12px; background: #fdfdfd;">
        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: bold; margin-bottom: 6px;">
          <span id="taskStatusMessage">Running task...</span>
          <span id="taskPercentText">0%</span>
        </div>
        <div style="width: 100%; height: 8px; background: #e2e2e0; border-radius: 4px; overflow: hidden;">
          <div id="taskProgressBar" style="width: 0%; height: 100%; background: var(--accent); transition: width 0.3s ease;"></div>
        </div>
      </div>

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
    const fetchMore = document.getElementById("fetchMore");
    const assetGrid = document.getElementById("assets");
    let assetList = [];
    let selectedAssetIds = [];

    let lastFetchSource = "";
    let fetchOffsets = { wikimedia: 0, pixabay: 0, ai: 0 };

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
        assetList = assetList.concat(data.assets);
        const remainingSpots = selectedVisualCount() - selectedAssetIds.length;
        if (remainingSpots > 0) {
          const newSelected = data.assets.slice(0, remainingSpots).map(asset => asset.id);
          selectedAssetIds = selectedAssetIds.concat(newSelected);
        }
        renderAssets();
        result.textContent = `Loaded ${data.assets.length} uploaded visuals. Total: ${assetList.length}.`;
      } catch (err) {
        result.textContent = String(err);
      } finally {
        uploadAssets.disabled = false;
        uploadAssets.textContent = "Upload Visuals";
      }
    });

    function pollTask(taskId, onProgress, onSuccess, onFailure) {
      const interval = setInterval(async () => {
        try {
          const res = await fetch(`/task/${taskId}`);
          if (!res.ok) throw new Error("Status check failed");
          const data = await res.json();
          
          if (data.status === "running" || data.status === "pending") {
            onProgress(data.percent, data.message, data.log);
          } else if (data.status === "completed") {
            clearInterval(interval);
            onSuccess(data.result);
          } else if (data.status === "failed") {
            clearInterval(interval);
            onFailure(data.message || "Task failed.", data.log);
          }
        } catch (err) {
          clearInterval(interval);
          onFailure(err.message || String(err));
        }
      }, 1000);
      return interval;
    }

    async function loadAssetsFromEndpoint(buttonEl, url, sourceKey, loadingText, doneText, isMore = false) {
      const topic = document.getElementById("topic").value.trim();
      if (!topic) {
        result.textContent = "Enter a topic first.";
        return;
      }
      
      const currentOffset = isMore ? fetchOffsets[sourceKey] : 0;
      if (!isMore) {
        fetchOffsets[sourceKey] = 0;
      }

      buttonEl.disabled = true;
      fetchMore.disabled = true;
      
      const progressContainer = document.getElementById("taskProgressContainer");
      const statusMsg = document.getElementById("taskStatusMessage");
      const percentText = document.getElementById("taskPercentText");
      const progressBar = document.getElementById("taskProgressBar");
      
      progressContainer.style.display = "block";
      statusMsg.textContent = loadingText;
      percentText.textContent = "0%";
      progressBar.style.width = "0%";

      try {
        const queryUrl = url + `?limit=6&start_offset=${currentOffset}&topic=${encodeURIComponent(topic)}`;
        const startRes = await fetch(queryUrl);
        const startData = await startRes.json();
        if (!startRes.ok || !startData.ok) {
          throw new Error(startData.message || "Failed to start task.");
        }

        const taskId = startData.task_id;
        pollTask(taskId, 
          (percent, msg) => {
            statusMsg.textContent = msg;
            percentText.textContent = percent + "%";
            progressBar.style.width = percent + "%";
          },
          (resData) => {
            progressContainer.style.display = "none";
            buttonEl.disabled = false;
            fetchMore.disabled = false;
            
            if (resData.assets && resData.assets.length > 0) {
              assetList = assetList.concat(resData.assets);
              const remainingSpots = selectedVisualCount() - selectedAssetIds.length;
              if (remainingSpots > 0) {
                const newSelected = resData.assets.slice(0, remainingSpots).map(asset => asset.id);
                selectedAssetIds = selectedAssetIds.concat(newSelected);
              }
              
              fetchOffsets[sourceKey] += resData.assets.length;
              lastFetchSource = sourceKey;
              
              renderAssets();
              result.textContent = `${doneText}: ${resData.assets.length} new visuals loaded. Total: ${assetList.length}.`;
              fetchMore.style.display = "block";
              fetchMore.textContent = `Fetch More (${sourceKey.toUpperCase()})`;
            } else {
              result.textContent = "No assets returned.";
            }
          },
          (errMsg) => {
            progressContainer.style.display = "none";
            buttonEl.disabled = false;
            fetchMore.disabled = false;
            result.textContent = "Error: " + errMsg;
          }
        );
      } catch (err) {
        progressContainer.style.display = "none";
        buttonEl.disabled = false;
        fetchMore.disabled = false;
        result.textContent = String(err);
      }
    }

    generateAiAssets.addEventListener("click", async () => {
      await loadAssetsFromEndpoint(generateAiAssets, "/generate-ai-assets", "ai", "Generating AI images...", "AI images ready");
      generateAiAssets.textContent = "Generate AI Images";
    });

    fetchPixabayAssets.addEventListener("click", async () => {
      await loadAssetsFromEndpoint(fetchPixabayAssets, "/fetch-pixabay-assets", "pixabay", "Fetching Pixabay...", "Pixabay visuals ready");
      fetchPixabayAssets.textContent = "Fetch From Pixabay";
    });

    fetchAssets.addEventListener("click", async () => {
      await loadAssetsFromEndpoint(fetchAssets, "/fetch-assets", "wikimedia", "Fetching Wikimedia...", "Wikimedia visuals ready");
      fetchAssets.textContent = "Fetch From Wikimedia";
    });

    fetchMore.addEventListener("click", () => {
      if (lastFetchSource === "wikimedia") {
        loadAssetsFromEndpoint(fetchAssets, "/fetch-assets", "wikimedia", "Fetching more Wikimedia...", "Wikimedia visuals ready", true);
      } else if (lastFetchSource === "pixabay") {
        loadAssetsFromEndpoint(fetchPixabayAssets, "/fetch-pixabay-assets", "pixabay", "Fetching more Pixabay...", "Pixabay visuals ready", true);
      } else if (lastFetchSource === "ai") {
        loadAssetsFromEndpoint(generateAiAssets, "/generate-ai-assets", "ai", "Generating more AI images...", "AI images ready", true);
      }
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

      const progressContainer = document.getElementById("taskProgressContainer");
      const statusMsg = document.getElementById("taskStatusMessage");
      const percentText = document.getElementById("taskPercentText");
      const progressBar = document.getElementById("taskProgressBar");
      
      progressContainer.style.display = "block";
      statusMsg.textContent = "Starting video generation...";
      percentText.textContent = "0%";
      progressBar.style.width = "0%";

      try {
        const res = await fetch("/generate", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            topic: document.getElementById("topic").value,
            manual_script: document.getElementById("manualScript").value,
            voice: document.getElementById("voice").value,
            use_music: document.getElementById("music").checked,
            visual_count: selectedVisualCount(),
            subtitle_position: document.getElementById("subtitlePosition").value,
            selected_visual_ids: selectedAssetIds
          })
        });
        const data = await res.json();
        if (!res.ok || !data.ok) {
          throw new Error(data.message || "Failed to start generation.");
        }

        const taskId = data.task_id;
        pollTask(taskId,
          (percent, msg, taskLog) => {
            statusMsg.textContent = msg;
            percentText.textContent = percent + "%";
            progressBar.style.width = percent + "%";
            if (taskLog) {
              log.textContent = taskLog;
              log.scrollTop = log.scrollHeight;
            }
          },
          (resData) => {
            progressContainer.style.display = "none";
            button.disabled = false;
            button.textContent = "Generate Short";
            
            result.textContent = resData.message;
            if (resData.log) {
              log.textContent = resData.log;
              log.scrollTop = log.scrollHeight;
            }
            if (resData.post) {
              post.textContent = `Title: ${resData.post.title}\n\nDescription:\n${resData.post.description}\n\nHashtags:\n${resData.post.hashtags}`;
            }
            if (resData.video_url) {
              video.src = resData.video_url;
              video.load();
            }
          },
          (errMsg, taskLog) => {
            progressContainer.style.display = "none";
            button.disabled = false;
            button.textContent = "Generate Short";
            result.textContent = "Generation failed.";
            if (taskLog) {
              log.textContent = taskLog;
            } else {
              log.textContent = errMsg;
            }
          }
        );
      } catch (err) {
        progressContainer.style.display = "none";
        button.disabled = false;
        button.textContent = "Generate Short";
        result.textContent = "Request failed.";
        log.textContent = String(err);
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
