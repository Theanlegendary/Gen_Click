import os
import re
import requests
import urllib.parse
import hashlib
from io import BytesIO
from PIL import Image, ImageDraw

IMAGE_WIDTH = 720
IMAGE_HEIGHT = 1280
IMAGE_MODEL = "sana"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
STOP_WORDS = {
    "about", "above", "after", "again", "against", "also", "and", "are", "because",
    "before", "between", "beyond", "but", "camera", "cinematic", "close", "color",
    "colors", "digital", "dramatic", "from", "have", "high", "image", "into", "lighting",
    "main", "more", "no", "not", "ratio", "scene", "shot", "style", "subject", "text",
    "the", "their", "this", "through", "vertical", "very", "with", "without", "watermark",
    "your", "fact", "facts", "top", "his", "her", "him", "she", "they",
}

def generate_image(prompt, index):
    """
    Sends the prompt to a 100% free image generation API (Pollinations.ai)
    """
    if index is not None:
        print(f"Generating image for scene {index}...")
    
    # URL encode the prompt
    encoded_prompt = urllib.parse.quote(prompt)
    
    # Pollinations AI image endpoint. The older /p/ URL can return HTML pages.
    query = urllib.parse.urlencode({
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "model": IMAGE_MODEL,
        "nologo": "true",
        "private": "true",
    })
    api_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?{query}"
    
    return api_url

import time
def download_image(image_url, output_filename="output.jpg", max_retries=3):
    """Downloads the generated image from the URL with retry logic."""
    print(f"Downloading image...")
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, headers=headers, timeout=90, allow_redirects=True)
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "").lower()
                if "image" not in content_type:
                    preview = response.text[:120].replace("\n", " ")
                    raise Exception(f"Expected image response but got '{content_type}': {preview}")

                image = Image.open(BytesIO(response.content))
                image.load()
                if image.width < 64 or image.height < 64:
                    raise Exception(f"Downloaded image is too small: {image.width}x{image.height}")

                os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
                output_ext = os.path.splitext(output_filename)[1].lower()
                if output_ext == ".png":
                    image.save(output_filename, "PNG", optimize=True)
                else:
                    image.convert("RGB").save(output_filename, "JPEG", quality=95)

                print(f"Image saved as {output_filename} ({image.width}x{image.height}, {image.format or content_type})")
                return os.path.abspath(output_filename)
            else:
                raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            print(f"   [Warning] Download attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise Exception(f"Failed to download image after {max_retries} attempts.")

def make_image_search_query(text, max_words=10):
    """
    Turns a script/caption/prompt into a short search query for fallback images.
    """
    cleaned = re.sub(r"[^A-Za-z0-9'\s-]", " ", text)
    words = []
    seen = set()
    for word in cleaned.split():
        normalized = word.strip("-'").lower()
        if len(normalized) < 3 or normalized in STOP_WORDS or normalized in seen:
            continue
        words.append(word.strip("-'"))
        seen.add(normalized)
        if len(words) >= max_words:
            break
    return " ".join(words) or "cinematic background"

def make_fallback_queries(text):
    """
    Builds several searches from most specific to broad main-subject searches.
    """
    query = make_image_search_query(text)
    queries = [query]
    lower_text = text.lower()

    proper_phrases = re.findall(r"\b[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*){1,3}\b", text)
    for phrase in proper_phrases:
        parts = [part for part in phrase.split() if part.lower() not in STOP_WORDS]
        if len(parts) >= 2:
            proper_query = " ".join(parts[:3])
            enriched_queries = []
            if "portrait" in lower_text:
                enriched_queries.append(f"{proper_query} portrait")
            if any(word in lower_text for word in ("photo", "realistic", "historical")):
                enriched_queries.append(f"{proper_query} photo")
            if "equipment" in lower_text:
                enriched_queries.append(f"{proper_query} equipment")
            if proper_query not in queries:
                queries.insert(0, proper_query)
            for enriched_query in reversed(enriched_queries):
                if enriched_query not in queries:
                    queries.insert(0, enriched_query)

    words = query.split()
    for length in (5, 4, 3, 2):
        shorter_query = " ".join(words[:length])
        if shorter_query and shorter_query not in queries:
            queries.append(shorter_query)

    return queries

def download_relevant_image(search_text, output_filename="fallback.png", max_results=8):
    """
    Downloads a relevant public-domain/Creative-Commons style image from Wikimedia Commons.
    Used only when AI image generation fails.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    candidates = []
    tried_queries = make_fallback_queries(search_text)
    for query in tried_queries:
        print(f"Searching downloadable fallback image: {query}")
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": str(max_results),
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "iiurlwidth": str(IMAGE_WIDTH),
            "format": "json",
        }
        response = requests.get(COMMONS_API_URL, params=params, headers=headers, timeout=20)
        response.raise_for_status()

        pages = (response.json().get("query") or {}).get("pages", {})
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            mime = (info.get("mime") or "").lower()
            url = info.get("thumburl") or info.get("url")
            if url and mime in {"image/jpeg", "image/png"} and url not in candidates:
                candidates.append(url)

        if candidates:
            break

    if not candidates:
        raise Exception(f"No downloadable image found for: {', '.join(tried_queries)}")

    last_error = None
    for candidate_url in candidates:
        try:
            return download_image(candidate_url, output_filename, max_retries=1)
        except Exception as e:
            last_error = e

    raise Exception(f"Fallback image download failed: {last_error}")

def generate_gradient_fallback(prompt, output_filename):
    """
    Generates a beautiful vibrant gradient background based on the prompt's hash
    so that it is deterministic and looks highly styled (GenZ aesthetic).
    """
    palettes = [
        ((138, 43, 226), (255, 20, 147)),  # Violet to Pink
        ((0, 139, 139), (65, 105, 225)),    # Cyan to Royal Blue
        ((255, 69, 0), (255, 165, 0)),     # Orange to Yellow
        ((75, 0, 130), (0, 128, 128)),      # Purple to Teal
        ((0, 191, 255), (199, 21, 133))    # Blue to Magenta
    ]
    # Select palette based on prompt hash
    hash_val = int(hashlib.md5(prompt.encode('utf-8')).hexdigest(), 16)
    color_start, color_end = palettes[hash_val % len(palettes)]
    
    # Create image
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT))
    draw = ImageDraw.Draw(img)
    
    # Generate vertical gradient
    for y in range(IMAGE_HEIGHT):
        ratio = y / IMAGE_HEIGHT
        r = int(color_start[0] * (1 - ratio) + color_end[0] * ratio)
        g = int(color_start[1] * (1 - ratio) + color_end[1] * ratio)
        b = int(color_start[2] * (1 - ratio) + color_end[2] * ratio)
        draw.line([(0, y), (IMAGE_WIDTH, y)], fill=(r, g, b))
        
    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
    img.save(output_filename, "PNG", optimize=True)
    print(f"   [Fallback] Generated gradient image for failed prompt: {output_filename}")
    return os.path.abspath(output_filename)

if __name__ == "__main__":
    url = generate_image("A cartoon cat in Korea", 1)
    download_image(url, "test_image.jpg")
