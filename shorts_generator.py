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
    print("[Error] Valid GEMINI_API_KEY not found in .env file.")
    print("Please get a free key from Google AI Studio and put it in your .env file.")
    sys.exit(1)

from PIL import Image
from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip, VideoFileClip

# Import our custom utilities
import audio_utils
import subtitle_utils
from image_generator import download_image, download_relevant_image, generate_image, generate_gradient_fallback

VIDEO_WIDTH = 720
VIDEO_HEIGHT = 1280
TARGET_SHORT_DURATION = 45.0
VOICE_RATE = "+20%"
FAST_IMAGE_DOWNLOAD_MODE = False
FAST_STATIC_IMAGES = True
SCRIPT_TIMEOUT_SECONDS = 25
IMAGE_WORKERS = 5
SCENE_COUNT = 5
RENDER_FPS = 24
ENCODER_PRESET = "ultrafast"
USE_PEXELS_VIDEOS = True
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

def count_spoken_words(text):
    words = re.findall(r"[A-Za-z0-9']+", text)
    return max(len(words), 1)

def generate_script_and_storyboard(topic, max_duration=TARGET_SHORT_DURATION):
    """
    Uses Gemini 1.5 Flash with structured JSON output to write a viral script and storyboard.
    """
    print("\n1. Writing viral script and visual scene prompts using Gemini...")
    max_duration = TARGET_SHORT_DURATION
    target_min_words = 95
    target_max_words = 145
    
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
        f"- Target a fast-paced YouTube Short between 30 and 60 seconds.\n"
        f"- The total script (combining all 'speech' fields) should be {target_min_words}-{target_max_words} words.\n"
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
        # Simple fallback template
        return [
            {
                "speech": f"Did you know the secret of {topic}? It is more shocking than you think.",
                "scene_prompt": f"Dramatic glowing concept art representing {topic}, vertical 9:16 ratio, cinematic lighting"
            },
            {
                "speech": "People have searched for answers for centuries, but they were looking in the wrong direction.",
                "scene_prompt": "Ancient mystery explorer looking at ancient ruins, cinematic lighting, digital art, vertical 9:16"
            },
            {
                "speech": "The truth is hidden right in front of us, waiting to be discovered.",
                "scene_prompt": "A glowing key lying on a desk, soft dust particles in sun rays, realistic, vertical 9:16"
            },
            {
                "speech": "Follow for more mind-blowing secrets and comment what you want to see next.",
                "scene_prompt": "A modern futuristic sign with follow icon glowing, dark aesthetic, cinematic, vertical 9:16"
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
    start_zoom = 1.04 if zoom_in else 1.10
    end_zoom = 1.10 if zoom_in else 1.04

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

async def create_shorts_video(topic, voice_name="male", use_bg_music=True, max_duration=TARGET_SHORT_DURATION):
    max_duration = TARGET_SHORT_DURATION
    
    print("\n" + "="*50)
    print(f"  CREATING YOUTUBE SHORT: {topic.upper()}")
    print(f"  TARGET LENGTH: {max_duration:.0f} SECONDS")
    print("="*50)
    
    # 1. Generate script & scene prompts
    segments = generate_script_and_storyboard(topic, max_duration)
    
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
    grouped_captions = subtitle_utils.group_words(words, max_chars=22, max_words=4)
    ass_path = os.path.join(folder_name, "subtitles.ass")
    subtitle_utils.generate_ass_file(words, grouped_captions, ass_path)
    
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
    
    print("\n" + "="*50)
    print("  [SUCCESS] YOUTUBE SHORT VIDEO GENERATED SUCCESSFULLY!")
    print(f"  Saved video path: {os.path.abspath(output_video_path)}")
    print("="*50 + "\n")
    return os.path.abspath(output_video_path)

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
