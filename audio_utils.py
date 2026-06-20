import os
import re
import random
import requests
import asyncio
import edge_tts

# Default realistic English voices
DEFAULT_VOICES = {
    "male": "en-US-ChristopherNeural",
    "female": "en-US-JennyNeural",
    "child": "en-US-AnaNeural"
}

async def generate_speech_and_srt(text, output_audio_path, voice_name="male", rate="+20%"):
    """
    Generates text-to-speech audio and captures word-level boundaries (SRT) in a single stream.
    """
    voice = DEFAULT_VOICES.get(voice_name, voice_name)
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    submaker = edge_tts.SubMaker()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
    
    # Stream the audio chunks and feed the boundaries to the submaker
    with open(output_audio_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                f.write(chunk["data"])
            elif chunk.get("type") in ("WordBoundary", "SentenceBoundary"):
                submaker.feed(chunk)
                
    # Get standard SRT formatted subtitles
    srt_content = submaker.get_srt()
    
    # Write SRT to a file next to the audio
    srt_path = os.path.splitext(output_audio_path)[0] + ".srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
        
    return srt_path

def parse_srt(srt_content):
    """
    Parses SRT format into a clean list of dicts: [{'word': '...', 'start': 0.1, 'end': 0.5}]
    """
    # Normalize line endings
    srt_content = srt_content.replace("\r\n", "\n")
    
    sentences = []
    # Pattern to match SRT blocks
    pattern = re.compile(
        r"(\d+)\s*\n"
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})\s*\n"
        r"(.*?)(?=\n\n|\n*$|\Z)",
        re.DOTALL
    )
    
    def to_seconds(hours, minutes, seconds, milliseconds):
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0
        
    for match in pattern.finditer(srt_content):
        idx, h1, m1, s1, ms1, h2, m2, s2, ms2, word_text = match.groups()
        start_time = to_seconds(h1, m1, s1, ms1)
        end_time = to_seconds(h2, m2, s2, ms2)
        sentences.append({
            "text": word_text.strip().replace('\n', ' '),
            "start": start_time,
            "end": end_time
        })
        
    return sentences

def interpolate_words(sentence_list):
    """
    Takes sentence boundaries and interpolates them into word timings.
    """
    word_list = []
    for item in sentence_list:
        text = item["text"]
        start = item["start"]
        end = item["end"]
        duration = end - start
        
        words = text.split()
        if not words:
            continue
            
        word_duration = duration / len(words)
        
        for idx, w in enumerate(words):
            w_start = start + idx * word_duration
            w_end = w_start + word_duration
            word_list.append({
                "word": w,
                "start": w_start,
                "end": w_end
            })
            
    return word_list

def get_srt_words(srt_file_path):
    """
    Helper to read an SRT file and return the parsed words.
    """
    with open(srt_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    sentences = parse_srt(content)
    # If the TTS returned SentenceBoundary, we interpolate into words.
    return interpolate_words(sentences)

def download_random_bg_music(output_dir="music"):
    """
    Fetches the catalog.json from open-lofi on GitHub, picks a random track, 
    and downloads it to use as background music.
    """
    print("Selecting background music...")
    os.makedirs(output_dir, exist_ok=True)
    
    catalog_url = "https://raw.githubusercontent.com/btahir/open-lofi/main/catalog.json"
    
    try:
        response = requests.get(catalog_url, timeout=10)
        response.raise_for_status()
        catalog = response.json()
        
        tracks = catalog.get("tracks", [])
        if not tracks:
            raise ValueError("No tracks found in the open-lofi catalog.")
            
        track = random.choice(tracks)
        category = track.get("category")
        filename = track.get("filename")
        title = track.get("title")
        
        encoded_filename = requests.utils.quote(filename)
        download_url = f"https://raw.githubusercontent.com/btahir/open-lofi/main/{category}/{encoded_filename}"
        
        output_file = os.path.join(output_dir, filename)
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 100000:
            print(f"Using cached music: '{title}' ({category})")
            return os.path.abspath(output_file)
            
        print(f"Downloading royalty-free music: '{title}' ({category})...")
        music_response = requests.get(download_url, timeout=30)
        music_response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(music_response.content)
            
        print(f"Music saved to: {output_file}")
        return os.path.abspath(output_file)
        
    except Exception as e:
        print(f"Warning: Failed to fetch online music ({e}). Looking for local fallback...")
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith(".mp3")]
            if files:
                fallback_file = os.path.join(output_dir, files[0])
                print(f"Found local music fallback: {fallback_file}")
                return os.path.abspath(fallback_file)
                
        print("No music files available. Proceeding without background music.")
        return None

if __name__ == "__main__":
    async def test():
        print("Testing edge-tts and music downloader...")
        audio_file = "test_audio.mp3"
        srt_file = await generate_speech_and_srt(
            "Hello world! This is a test of sentence boundaries.", 
            audio_file, 
            "male"
        )
        print(f"Speech saved, SRT at {srt_file}")
        words = get_srt_words(srt_file)
        print(f"Parsed {len(words)} words. First word: {words[0]}")
        
        music_file = download_random_bg_music("test_music")
        print(f"Music downloaded at: {music_file}")
        
    asyncio.run(test())
