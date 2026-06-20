import os
import sys
import json
import time
import asyncio
import re
import subprocess
import shutil
import imageio_ffmpeg
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("[Warning] Valid GEMINI_API_KEY not found in .env file.")
    print("Add a free key from Google AI Studio before generating videos.")

from PIL import Image
from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip, VideoFileClip

# Import our custom utilities
import audio_utils
import subtitle_utils
from image_generator import download_image, download_relevant_image, generate_image, generate_gradient_fallback

VIDEO_WIDTH = 720
VIDEO_HEIGHT = 1280
TARGET_SHORT_DURATION = 60.0
VOICE_RATE = "+20%"
FAST_IMAGE_DOWNLOAD_MODE = False
FAST_STATIC_IMAGES = False
SCRIPT_TIMEOUT_SECONDS = 25
IMAGE_WORKERS = 5
SCENE_COUNT = 12
RENDER_FPS = 24
ENCODER_PRESET = "ultrafast"
USE_PEXELS_VIDEOS = True
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
SELECTED_VISUAL_PATHS = []

def count_spoken_words(text):
    words = re.findall(r"[A-Za-z0-9']+", text)
    return max(len(words), 1)

def build_final_output_path(topic):
    clean_topic = "".join([c if c.isalnum() else "_" for c in topic]).strip("_")[:32]
    clean_topic = clean_topic or "short"
    return os.path.abspath(f"final_short_{int(time.time())}_{clean_topic}.mp4")

def generate_script_and_storyboard(topic, max_duration=TARGET_SHORT_DURATION):
    """
    Uses Gemini 1.5 Flash with structured JSON output to write a viral script and storyboard.
    """
    print("\n1. Writing viral script and visual scene prompts using Gemini...")
    max_duration = TARGET_SHORT_DURATION
    target_min_words = 185
    target_max_words = 220
    
    # We use gemini-2.5-flash which is fast, high quality, and supports structured JSON outputs
    generation_config = {
        "response_mime_type": "application/json"
    }
    model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)
    
    prompt = (
        f"Create an engaging YouTube Shorts script and visual storyboard about the topic: '{topic}'.\n"
        "Return the output STRICTLY as a JSON array of objects. Each object MUST have two keys:\n"
        "1. 'speech': A 1-2 sentence narration that is spoken, punchy, dramatic, and optimized for a voiceover. "
        "Use engaging hooks and words.\n"
        "2. 'scene_prompt': A highly detailed prompt for an image generator to illustrate this segment. "
        "Specify one clear main subject, camera angles, colors, high contrast, and include keywords like "
        "'9:16 vertical ratio', 'cinematic lighting', 'digital art style', 'no text', 'no watermark'.\n\n"
        "Constraints:\n"
        f"- The total list should contain exactly {SCENE_COUNT} objects (scenes).\n"
        f"- Target a fast-paced YouTube Short close to 60 seconds.\n"
        f"- The total script (combining all 'speech' fields) should be {target_min_words}-{target_max_words} words.\n"
        "- Write for a discovery/science/history audience. Make every scene teach one useful fact.\n"
        "- Avoid generic filler like 'the truth is hidden'. Give concrete details, stakes, and visual moments.\n"
        "- The first segment must start with a very strong, attention-grabbing hook.\n"
        "- The final segment must conclude with a loop hook or call to action."
    )
    
    try:
        response = model.generate_content(
            prompt,
            request_options={"timeout": SCRIPT_TIMEOUT_SECONDS}
        )
        text_output = response.text.strip()
        
        # Parse JSON
        segments = json.loads(text_output)
        if not isinstance(segments, list) or len(segments) == 0:
            raise ValueError("Response is not a valid list.")
            
        print(f"   Successfully generated {len(segments)} storyboard segments!")
        for idx, seg in enumerate(segments):
            print(f"     Segment {idx+1}: {len(seg['speech'].split())} words | Prompt: {seg['scene_prompt'][:50]}...")
            
        return segments
    except Exception as e:
        print(f"[Error] Failed to generate script via Gemini: {e}")
        print("Falling back to a template script...")
        return build_fallback_segments(topic)

def normalize_segment_count(segments, target_count):
    target_count = max(3, min(int(target_count), 15))
    segments = [dict(seg) for seg in segments if seg.get("speech") and seg.get("scene_prompt")]
    if not segments:
        return build_fallback_segments("world discovery")[:target_count]

    while len(segments) > target_count:
        segments.pop()

    while len(segments) < target_count:
        split_index = max(range(len(segments)), key=lambda idx: len(segments[idx]["speech"].split()))
        speech_words = segments[split_index]["speech"].split()
        if len(speech_words) < 12:
            clone = dict(segments[-1])
            clone["scene_prompt"] = f"{clone['scene_prompt']}, alternate angle, cinematic variation"
            segments.append(clone)
            continue

        midpoint = len(speech_words) // 2
        first = dict(segments[split_index])
        second = dict(segments[split_index])
        first["speech"] = " ".join(speech_words[:midpoint])
        second["speech"] = " ".join(speech_words[midpoint:])
        second["scene_prompt"] = f"{second['scene_prompt']}, alternate angle, new composition"
        segments[split_index:split_index + 1] = [first, second]

    return segments

def build_fallback_segments(topic):
    lower_topic = topic.lower()
    if "ocean" in lower_topic and ("sunlight" in lower_topic or "zone" in lower_topic):
        return [
            {
                "speech": "There is a line in the ocean where sunlight starts to lose the fight.",
                "scene_prompt": "deep ocean sunlight fading into darkness, vertical documentary shot, particles in blue water, cinematic, 9:16, no text"
            },
            {
                "speech": "Above that line, plankton use sunlight like tiny power stations. But deeper down, colors vanish one by one.",
                "scene_prompt": "underwater color spectrum disappearing with depth, diver descending, realistic ocean science documentary, vertical 9:16"
            },
            {
                "speech": "Red disappears first, then orange, then yellow, until blue is almost all that remains.",
                "scene_prompt": "deep sea diver watching red orange and yellow light fade into blue darkness, vertical 9:16"
            },
            {
                "speech": "Around two hundred meters down, you enter the twilight zone.",
                "scene_prompt": "ocean twilight zone with faint blue light and silhouettes of fish, cinematic lighting, vertical 9:16, no watermark"
            },
            {
                "speech": "There is still a little light there, but not enough for plants to feed the food chain.",
                "scene_prompt": "faint blue ocean twilight with sparse plankton and fish silhouettes, realistic documentary, 9:16"
            },
            {
                "speech": "By about one thousand meters, sunlight is basically gone.",
                "scene_prompt": "black deep ocean at one thousand meters with tiny distant glowing animals, vertical 9:16"
            },
            {
                "speech": "This is the midnight zone, where pressure is crushing and animals create their own light.",
                "scene_prompt": "midnight zone deep sea animals glowing with bioluminescence, dark ocean, high contrast, vertical 9:16, no text"
            },
            {
                "speech": "That glow is called bioluminescence, and it is one of nature's smartest survival tools.",
                "scene_prompt": "bioluminescent jellyfish and deep sea fish glowing in black water, macro documentary style, vertical 9:16"
            },
            {
                "speech": "Some creatures use it to hunt, some use it to hide, and some use it like a signal.",
                "scene_prompt": "glowing anglerfish and jellyfish using light signals in black water, documentary macro, vertical 9:16"
            },
            {
                "speech": "The strange part is that this dark world helps the whole planet breathe.",
                "scene_prompt": "deep ocean ecosystem carrying carbon into darkness, scientific cinematic visual, vertical 9:16"
            },
            {
                "speech": "Carbon sinks from the surface, carrying energy downward like a slow underwater snowfall.",
                "scene_prompt": "marine snow drifting through dark deep ocean with tiny particles, scientific documentary visual, vertical 9:16"
            },
            {
                "speech": "So when you see calm blue water, remember: below it is a hidden night world. Follow Discover World by Codex.",
                "scene_prompt": "surface ocean splitting into deep dark world below, explorer silhouette, epic discovery ending, vertical 9:16, no text"
            }
        ]

    return [
        {
            "speech": f"What if {topic} is not just a fact, but a clue about how our world really works?",
            "scene_prompt": f"cinematic discovery scene about {topic}, dramatic map table, glowing evidence, 9:16 vertical ratio, no text, no watermark"
        },
        {
            "speech": "Every discovery starts when someone notices one detail that other people ignore.",
            "scene_prompt": f"close up of overlooked evidence about {topic}, macro documentary style, vertical 9:16"
        },
        {
            "speech": "The useful question is not only what happened. The useful question is why it happened.",
            "scene_prompt": f"researcher connecting evidence about {topic} across maps and photos, cinematic lighting, realistic discovery style, vertical 9:16"
        },
        {
            "speech": "The cause usually connects nature, time, pressure, climate, and human history.",
            "scene_prompt": f"earth systems connecting climate time and history around {topic}, cinematic documentary, vertical 9:16"
        },
        {
            "speech": "Scientists and explorers often begin with something small: a strange rock, a pattern, or a shadow.",
            "scene_prompt": f"macro clues of rock ocean wall and satellite shadow linked to {topic}, documentary montage, vertical 9:16"
        },
        {
            "speech": "Then the hidden system appears, and one place starts to explain another.",
            "scene_prompt": f"earth systems and ancient places connected to {topic}, cinematic documentary, vertical 9:16, no text"
        },
        {
            "speech": "Weather changes landscapes. Water shapes cities. Ancient choices leave evidence behind.",
            "scene_prompt": f"weather water cities and ancient evidence connected to {topic}, realistic discovery montage, 9:16"
        },
        {
            "speech": "That is why discovery is powerful. It turns ordinary places into evidence.",
            "scene_prompt": f"dynamic earth landscape changing over time around {topic}, time-lapse style, dramatic light, vertical 9:16"
        },
        {
            "speech": "It also shows that Earth is not static. It is moving, changing, and recording its own story.",
            "scene_prompt": f"planet earth changing over time with geological and historical layers around {topic}, vertical 9:16"
        },
        {
            "speech": "When you understand the pattern, the world becomes more interesting.",
            "scene_prompt": f"person discovering hidden pattern in nature and history about {topic}, cinematic, vertical 9:16"
        },
        {
            "speech": "A desert, mountain, forest, ruin, storm, animal, or star can suddenly explain something bigger.",
            "scene_prompt": f"desert mountain forest ruin storm animal and stars as discovery clues for {topic}, cinematic vertical 9:16"
        },
        {
            "speech": "This is Discover World by Codex. Follow for hidden stories from Earth, science, history, nature, oceans, and space.",
            "scene_prompt": f"epic discovery channel style ending shot for {topic}, earth horizon, explorer silhouette, cinematic, vertical 9:16, no text"
        }
    ]

def process_image_to_vertical(input_path, output_path):
    """
    Crops any image to 9:16 ratio and resizes it to 1080x1920 for YouTube Shorts.
    """
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        target_aspect = 9 / 16
        current_aspect = w / h
        
        # Crop to 9:16 (vertical center-crop)
        if current_aspect > target_aspect:
            new_w = int(h * target_aspect)
            left = (w - new_w) // 2
            img_cropped = img.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_aspect)
            top = (h - new_h) // 2
            img_cropped = img.crop((0, top, w, top + new_h))
            
        # Resize to the configured vertical Shorts size.
        img_resized = img_cropped.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
        img_resized.save(output_path, "PNG", optimize=True)
    return output_path

def scale_word_timings(words, speed_factor):
    """
    Keeps captions and image timing synchronized after speeding up narration.
    """
    if speed_factor <= 1.0:
        return words

    scaled_words = []
    for word in words:
        scaled_word = dict(word)
        scaled_word["start"] = word["start"] / speed_factor
        scaled_word["end"] = word["end"] / speed_factor
        scaled_words.append(scaled_word)
    return scaled_words

def align_segments_with_words(segments, words, video_duration):
    """
    Aligns storyboard images to the actual voiceover timeline.
    Boundaries are chosen by proportional word position, then read from SRT timestamps.
    """
    aligned_segments = []
    script_word_counts = [count_spoken_words(seg["speech"]) for seg in segments]
    total_script_words = sum(script_word_counts) or 1
    current_start = 0.0
    cumulative_words = 0
    srt_word_count = len(words)

    for i, seg in enumerate(segments):
        start_time = current_start
        cumulative_words += script_word_counts[i]

        if i == len(segments) - 1 or not words:
            end_time = video_duration
        else:
            word_position = cumulative_words / total_script_words
            end_idx = round(word_position * srt_word_count) - 1
            end_idx = min(max(end_idx, 0), srt_word_count - 1)
            end_time = min(words[end_idx]["end"], video_duration)
            end_time = max(end_time, start_time + 0.25)
            end_time = min(end_time, video_duration)

        duration = max(end_time - start_time, 0.1)
        aligned_segments.append({
            "speech": seg["speech"],
            "scene_prompt": seg["scene_prompt"],
            "start": start_time,
            "end": end_time,
            "duration": duration
        })
        current_start = end_time
        
    return aligned_segments

def create_scene_clip(img_path, duration, start, index):
    """
    Creates a scene clip (supporting static ImageClips and stock VideoFileClips).
    """
    if img_path.endswith(".mp4"):
        return VideoFileClip(img_path).with_start(start).with_duration(duration)

    if FAST_STATIC_IMAGES:
        return ImageClip(img_path).with_duration(duration).with_start(start)

    zoom_in = index % 2 == 0
    start_zoom = 1.015 if zoom_in else 1.045
    end_zoom = 1.045 if zoom_in else 1.015

    def zoom(t):
        progress = min(max(t / max(duration, 0.01), 0), 1)
        return start_zoom + ((end_zoom - start_zoom) * progress)

    return (
        ImageClip(img_path)
        .with_duration(duration)
        .resized(zoom)
        .with_position(("center", "center"))
        .with_start(start)
    )

def download_pexels_video(search_query, output_path, duration):
    """
    Searches Pexels for a vertical stock video matching search_query,
    downloads it, and crops/trims it to the scene duration.
    """
    import requests
    if not PEXELS_API_KEY or PEXELS_API_KEY == "your_pexels_api_key_here":
        raise ValueError("PEXELS_API_KEY is not set or valid in your .env file.")
        
    # Extract a clean, short keyword search query
    from image_generator import make_image_search_query
    clean_query = make_image_search_query(search_query, max_words=3)
    print(f"      Searching Pexels video for query: '{clean_query}'...")
    
    headers = {"Authorization": PEXELS_API_KEY}
    url = "https://api.pexels.com/videos/search"
    params = {
        "query": clean_query,
        "per_page": 5,
        "orientation": "portrait"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        videos = data.get("videos", [])
        if not videos:
            broad_query = " ".join(clean_query.split()[:2])
            if broad_query and broad_query != clean_query:
                print(f"      No results. Trying broader search query: '{broad_query}'...")
                params["query"] = broad_query
                response = requests.get(url, headers=headers, params=params, timeout=20)
                response.raise_for_status()
                data = response.json()
                videos = data.get("videos", [])
                
        if not videos:
            print("      No results. Trying general cosmic query...")
            params["query"] = "cosmic space"
            response = requests.get(url, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            videos = data.get("videos", [])
            
        if not videos:
            raise Exception("No videos found on Pexels.")
            
        best_link = None
        for v in videos:
            video_files = v.get("video_files", [])
            for vf in video_files:
                link = vf.get("link")
                width = vf.get("width") or 0
                height = vf.get("height") or 0
                file_type = vf.get("file_type") or ""
                
                if link and "mp4" in file_type.lower() and height > width:
                    best_link = link
                    break
            if best_link:
                break
                
        if not best_link and videos:
            for vf in videos[0].get("video_files", []):
                link = vf.get("link")
                file_type = vf.get("file_type") or ""
                if link and "mp4" in file_type.lower():
                    best_link = link
                    break
                    
        if not best_link:
            raise Exception("No suitable MP4 video link found.")
            
        print(f"      Downloading stock video file...")
        res = requests.get(best_link, timeout=60, allow_redirects=True)
        res.raise_for_status()
        
        temp_full_path = output_path + "_temp.mp4"
        with open(temp_full_path, "wb") as f:
            f.write(res.content)
            
        print(f"      Cropping and trimming video to {duration:.2f}s...")
        clip = VideoFileClip(temp_full_path)
        
        if clip.duration > duration:
            clip = clip.subclipped(0, duration)
        else:
            clip = clip.with_duration(duration)
            
        w, h = clip.size
        target_w, target_h = VIDEO_WIDTH, VIDEO_HEIGHT
        
        aspect_ratio = target_w / target_h
        current_aspect = w / h
        
        if current_aspect > aspect_ratio:
            new_w = int(h * aspect_ratio)
            left = (w - new_w) // 2
            clip_cropped = clip.cropped(x1=left, y1=0, x2=left+new_w, y2=h)
        else:
            new_h = int(w / aspect_ratio)
            top = (h - new_h) // 2
            clip_cropped = clip.cropped(x1=0, y1=top, x2=w, y2=top+new_h)
            
        clip_final = clip_cropped.resized(new_size=(target_w, target_h))
        clip_final.write_videofile(output_path, fps=24, codec="libx264", audio=False, logger=None)
        
        clip.close()
        clip_final.close()
        if os.path.exists(temp_full_path):
            os.remove(temp_full_path)
            
        print(f"      Video saved successfully at: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"      [Warning] Pexels download failed: {e}")
        temp_full_path = output_path + "_temp.mp4"
        if os.path.exists(temp_full_path):
            os.remove(temp_full_path)
        raise e

def create_scene_image(topic, seg, idx, total_segments, folder_name):
    """
    Builds one scene clip. Try downloading Pexels stock video first if USE_PEXELS_VIDEOS is enabled.
    Falls back to generating custom AI images or gradients.
    """
    print(f"   Scene {idx+1}/{total_segments} (Duration: {seg['duration']:.2f}s)...")
    final_video_path = os.path.join(folder_name, f"scene_{idx+1}.mp4")
    temp_img_path = os.path.join(folder_name, f"temp_scene_{idx+1}.png")
    final_img_path = os.path.join(folder_name, f"scene_{idx+1}.png")

    if SELECTED_VISUAL_PATHS:
        selected_path = SELECTED_VISUAL_PATHS[idx % len(SELECTED_VISUAL_PATHS)]
        print(f"      Using selected visual: {selected_path}")
        if selected_path.lower().endswith(".mp4"):
            return selected_path
        process_image_to_vertical(selected_path, final_img_path)
        return final_img_path
    
    if USE_PEXELS_VIDEOS and PEXELS_API_KEY and PEXELS_API_KEY != "your_pexels_api_key_here":
        try:
            search_text = f"{topic} {seg['speech']}"
            return download_pexels_video(search_text, final_video_path, seg["duration"])
        except Exception as pexels_err:
            print(f"      [Warning] Stock video fallback to AI image generation: {pexels_err}")
            
    fallback_query = f"{topic} {seg['speech']} {seg['scene_prompt']}"
    try:
        if FAST_IMAGE_DOWNLOAD_MODE:
            print("      Downloading fast relevant image...")
            download_relevant_image(fallback_query, temp_img_path)
        else:
            image_url = generate_image(seg["scene_prompt"], idx + 1)
            download_image(image_url, temp_img_path)

        process_image_to_vertical(temp_img_path, final_img_path)
        return final_img_path
    except Exception as e:
        print(f"      [Warning] Primary image failed for scene {idx+1}: {e}")

        if not FAST_IMAGE_DOWNLOAD_MODE:
            try:
                print("      Trying downloaded fallback image from the script/caption...")
                download_relevant_image(fallback_query, temp_img_path)
                process_image_to_vertical(temp_img_path, final_img_path)
                return final_img_path
            except Exception as fallback_error:
                print(f"      [Warning] Downloaded fallback failed for scene {idx+1}: {fallback_error}")

        print("      Using gradient background fallback.")
        generate_gradient_fallback(seg["scene_prompt"], final_img_path)
        return final_img_path
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

async def create_shorts_video(topic, voice_name="male", use_bg_music=True, max_duration=TARGET_SHORT_DURATION, subtitle_position="middle"):
    max_duration = TARGET_SHORT_DURATION
    
    print("\n" + "="*50)
    print(f"  CREATING YOUTUBE SHORT: {topic.upper()}")
    print(f"  TARGET LENGTH: {max_duration:.0f} SECONDS")
    print("="*50)
    
    # 1. Generate script & scene prompts
    segments = normalize_segment_count(generate_script_and_storyboard(topic, max_duration), SCENE_COUNT)
    
    # Create story workspace directory
    clean_topic = "".join([c if c.isalnum() else "_" for c in topic]).strip("_")[:20]
    folder_name = f"short_{int(time.time())}_{clean_topic}"
    os.makedirs(folder_name, exist_ok=True)
    print(f"Workspace folder: {folder_name}")
    
    # Combine script speech to a single narrative
    full_script = " ".join([seg["speech"] for seg in segments])
    print(f"\nFull Script ({len(full_script.split())} words):\n{full_script}\n")
    
    # 2. Generate Audio & Subtitles
    print("2. Synthesizing voiceover and extracting word boundaries...")
    audio_path = os.path.join(folder_name, "voiceover.mp3")
    srt_path = await audio_utils.generate_speech_and_srt(full_script, audio_path, voice_name, rate=VOICE_RATE)
    words = audio_utils.get_srt_words(srt_path)
    original_voice_duration = words[-1]["end"] if words else 0.0
    speed_factor = 1.0
    print(f"   Audio generated at {VOICE_RATE} rate ({original_voice_duration:.2f} seconds) without speed-scaling. Timestamps parsed successfully!")

    words = scale_word_timings(words, speed_factor)
    
    # Align scenes to audio timing
    video_duration = original_voice_duration if original_voice_duration else max_duration
    aligned_segments = align_segments_with_words(segments, words, video_duration)
    print(f"   Final video duration: {video_duration:.2f} seconds")
    
    # 3. Generate and Process Images
    image_mode = "fast downloaded images" if FAST_IMAGE_DOWNLOAD_MODE else "AI images with downloaded fallback"
    print(f"\n3. Preparing visual scenes ({image_mode}, sequential)...")
    processed_images = [None] * len(aligned_segments)

    for idx, seg in enumerate(aligned_segments):
        if idx > 0:
            time.sleep(0.5)
        processed_images[idx] = create_scene_image(topic, seg, idx, len(aligned_segments), folder_name)
            
    # 4. Create Video Slideshow Clips
    print("\n4. Assembling background video clips...")
    image_clips = []
    for idx, seg in enumerate(aligned_segments):
        img_path = processed_images[idx]
        clip = create_scene_clip(img_path, seg["duration"], seg["start"], idx)
        image_clips.append(clip)
        
    background_video = CompositeVideoClip(image_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT)).with_duration(video_duration)
    
    # 5. Create Subtitles using ASS
    print("\n5. Generating dynamic ASS subtitles...")
    grouped_captions = subtitle_utils.group_words(words, max_chars=28, max_words=6)
    ass_path = os.path.join(folder_name, "subtitles.ass")
    subtitle_utils.generate_ass_file(words, grouped_captions, ass_path, position=subtitle_position)
    
    # 6. Setup Audio Track
    print("\n6. Mixing narration audio and background music...")
    raw_voice_audio = AudioFileClip(audio_path)
    voice_audio = raw_voice_audio.with_speed_scaled(factor=speed_factor) if speed_factor > 1.0 else raw_voice_audio
    voice_audio = voice_audio.with_duration(video_duration)
    
    audio_clips = [voice_audio]
    
    if use_bg_music:
        music_path = audio_utils.download_random_bg_music()
        if music_path:
            bg_music = AudioFileClip(music_path)
            
            # Crop background music to fit the final video length
            if bg_music.duration > video_duration:
                bg_music = bg_music.subclipped(0, video_duration)
            else:
                bg_music = bg_music.with_duration(video_duration)
                
            # Duck music volume to 12% so narrative is loud and clear
            bg_music = bg_music.with_volume_scaled(0.12)
            audio_clips.append(bg_music)
            
    composite_audio = CompositeAudioClip(audio_clips).with_duration(video_duration)
    
    # 7. Compile Clean Video slideshow (extremely fast without subtitle overlay processing)
    print("\n7. Rendering clean video slideshow (MoviePy)...")
    temp_clean_video_path = os.path.join(folder_name, "temp_clean_video.mp4")
    
    background_video = background_video.with_audio(composite_audio)
    
    background_video.write_videofile(
        temp_clean_video_path,
        fps=RENDER_FPS,
        codec="libx264",
        audio_codec="aac",
        preset=ENCODER_PRESET,
        threads=os.cpu_count() or 4,
        audio_bitrate="128k",
        pixel_format="yuv420p",
        logger=None  # Suppress verbose MoviePy output
    )
    
    # Close MoviePy resources before FFmpeg processing to release file locks
    voice_audio.close()
    if raw_voice_audio is not voice_audio:
        raw_voice_audio.close()
    composite_audio.close()
    for clip in image_clips:
        clip.close()
    background_video.close()

    # 8. Burn subtitles using native FFmpeg
    print("\n8. Burning dynamic active-word subtitles (native FFmpeg)...")
    output_video_path = os.path.join(folder_name, "final_shorts.mp4")
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Use relative path for FFmpeg filter on Windows to avoid drive letter colons
    safe_ass_path = ass_path.replace("\\", "/")
    
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", temp_clean_video_path,
        "-vf", f"subtitles={safe_ass_path}",
        "-c:v", "libx264",
        "-c:a", "copy",
        "-preset", "ultrafast",
        output_video_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"   [Error] FFmpeg subtitle burning failed: {e.stderr}")
        print("   Falling back to clean video without subtitles...")
        shutil.copy2(temp_clean_video_path, output_video_path)
    finally:
        # Clean up temporary clean video
        if os.path.exists(temp_clean_video_path):
            try:
                os.remove(temp_clean_video_path)
            except Exception as clean_err:
                print(f"   [Warning] Failed to clean up temp video file: {clean_err}")
    
    final_output_path = build_final_output_path(topic)
    shutil.copy2(output_video_path, final_output_path)

    try:
        shutil.rmtree(folder_name)
        print(f"  Cleaned temporary workspace: {folder_name}")
    except Exception as clean_err:
        print(f"   [Warning] Failed to clean up workspace folder {folder_name}: {clean_err}")

    print("\n" + "="*50)
    print("  [SUCCESS] YOUTUBE SHORT VIDEO GENERATED SUCCESSFULLY!")
    print(f"  Saved video path: {final_output_path}")
    print("="*50 + "\n")
    return final_output_path

if __name__ == "__main__":
    # Prompt the user for details
    print("==================================================")
    print("      AI Faceless YouTube Shorts Generator        ")
    print("==================================================")
    
    topic = input("Enter a topic for your YouTube Short (e.g., 'Psychology Secrets'): ").strip()
    if not topic:
        topic = "Shocking Universe Facts"
        print(f"No topic entered. Using default: '{topic}'")
        
    voice_choice = input("Choose voice - 'male', 'female', or 'child' [Default: male]: ").strip().lower()
    if voice_choice not in ["male", "female", "child"]:
        voice_choice = "male"
        
    music_choice = input("Do you want royalty-free background music? (y/n) [Default: n for fastest]: ").strip().lower()
    use_music = music_choice == 'y'
    
    # Run the generator
    asyncio.run(create_shorts_video(topic, voice_choice, use_music))
