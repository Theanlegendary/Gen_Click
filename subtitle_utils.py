import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Path to standard fonts on Windows
FONT_OPTIONS = [
    r"C:\Windows\Fonts\impact.ttf",      # Classic Shorts font
    r"C:\Windows\Fonts\arialbd.ttf",     # Arial Bold
    r"C:\Windows\Fonts\tahomabd.ttf",    # Tahoma Bold
    r"C:\Windows\Fonts\arial.ttf"        # Arial Regular fallback
]

def get_system_font(preferred_path=None):
    """
    Finds a suitable TrueType font on the system, falling back if necessary.
    """
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path
        
    for font_path in FONT_OPTIONS:
        if os.path.exists(font_path):
            return font_path
            
    return None  # Will fallback to PIL default font

def group_words(words, max_chars=15, max_words=3, max_pause=0.4):
    """
    Groups words into short phrases for captions.
    Breaks groups based on word count, character count, pauses, and punctuation.
    """
    if not words:
        return []
        
    groups = []
    current_group = []
    current_chars = 0
    
    for word in words:
        word_text = word["word"]
        word_start = word["start"]
        word_end = word["end"]
        
        # Check if we should split before adding this word
        should_split = False
        if current_group:
            # 1. Too many words
            if len(current_group) >= max_words:
                should_split = True
            # 2. Too many characters
            elif current_chars + len(word_text) > max_chars:
                should_split = True
            # 3. Pause in speech
            elif word_start - current_group[-1]["end"] > max_pause:
                should_split = True
            # 4. Punctuation in the previous word (sentence end)
            elif current_group[-1]["word"].endswith((".", "?", "!")):
                should_split = True
                
        if should_split:
            # Save the completed group
            groups.append({
                "text": " ".join([w["word"] for w in current_group]).upper(),
                "start": current_group[0]["start"],
                "end": current_group[-1]["end"]
            })
            current_group = [word]
            current_chars = len(word_text)
        else:
            current_group.append(word)
            current_chars += len(word_text) + 1  # +1 for space
            
    # Add the last remaining group
    if current_group:
        groups.append({
            "text": " ".join([w["word"] for w in current_group]).upper(),
            "start": current_group[0]["start"],
            "end": current_group[-1]["end"]
        })
        
    return groups

def render_subtitle_frame(text, font_path=None, font_size=80, text_color=(255, 242, 0, 255), stroke_color=(0, 0, 0, 255), stroke_width=8, width=1080, height=1920):
    """
    Renders text to a transparent video-sized image and returns it as a numpy array.
    """
    # Create RGBA transparent image
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Load font
    font = None
    resolved_font_path = get_system_font(font_path)
    if resolved_font_path:
        try:
            font = ImageFont.truetype(resolved_font_path, font_size)
        except Exception as e:
            print(f"Error loading TrueType font: {e}")
            
    if font is None:
        font = ImageFont.load_default()
        
    # Get text dimensions to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Position: Centered horizontally, 75% height from top (Y = 1440)
    x = (width - text_w) // 2
    y = int(height * 0.70) - (text_h // 2)
    
    # Draw text with high-impact stroke outline
    draw.text(
        (x, y), 
        text, 
        font=font, 
        fill=text_color, 
        stroke_width=stroke_width, 
        stroke_fill=stroke_color
    )
    
    # Return as numpy array for MoviePy
    return np.array(image)

def create_subtitle_clips(grouped_words, font_size=85, text_color=(255, 242, 0, 255), stroke_width=8, width=1080, height=1920):
    """
    Creates a list of MoviePy ImageClips with transparency, synchronized to the speech timings.
    """
    from moviepy import ImageClip
    
    clips = []
    font_path = get_system_font()
    
    print(f"Creating subtitle clips for {len(grouped_words)} phrases...")
    
    for group in grouped_words:
        text = group["text"]
        start_time = group["start"]
        end_time = group["end"]
        duration = end_time - start_time
        
        if duration <= 0:
            continue
            
        # Render the PIL image frame and convert to numpy array
        frame_array = render_subtitle_frame(
            text, 
            font_path=font_path, 
            font_size=font_size, 
            text_color=text_color,
            stroke_width=stroke_width,
            width=width,
            height=height
        )
        
        # In MoviePy 2.x, transparency is supported by passing the RGBA array.
        # We split the array into RGB and Alpha mask.
        rgb_array = frame_array[:, :, :3]
        alpha_mask = frame_array[:, :, 3] / 255.0
        
        # Create the image clip and apply mask
        clip = ImageClip(rgb_array).with_duration(duration).with_start(start_time)
        mask_clip = ImageClip(alpha_mask, is_mask=True).with_duration(duration).with_start(start_time)
        clip = clip.with_mask(mask_clip)
        
        clips.append(clip)
        
    return clips

def to_ass_timestamp(seconds):
    """
    Converts seconds into ASS timestamp format: H:MM:SS.CC
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    if centiseconds == 100:
        secs += 1
        centiseconds = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

def escape_ass_text(text):
    return str(text).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

def format_two_line_ass(words):
    if len(words) <= 3:
        return " ".join(words)

    midpoint = max(1, round(len(words) / 2))
    first = " ".join(words[:midpoint])
    second = " ".join(words[midpoint:])
    return f"{first}\\N{second}"

def generate_ass_file(words, grouped_captions, ass_path, subtitle_offset=-0.15, position="middle"):
    """
    Generates an ASS subtitle file with dynamic active-word highlighting.
    """
    position_styles = {
        "up": (8, 90),
        "middle": (5, 0),
        "bottom": (2, 120),
    }
    alignment, margin_v = position_styles.get(str(position).lower(), position_styles["middle"])

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 720\n")
        f.write("PlayResY: 1280\n\n")
        
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        # Text is split into no more than two lines. Alignment controls top/middle/bottom.
        f.write(f"Style: Default,Impact,52,&H00FFFFFF,&H0000F2FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,4,1,{alignment},40,40,{margin_v},1\n\n")
        
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        word_index = 0
        for group in grouped_captions:
            group_start = group["start"]
            group_end = group["end"]
            group_text_raw = group["text"]
            
            # Find all words that fall within this group's start and end time
            group_words = []
            while word_index < len(words) and words[word_index]["start"] < group_end:
                if words[word_index]["end"] > group_start:
                    clean_word = words[word_index].copy()
                    group_words.append(clean_word)
                word_index += 1
                
            if not group_words:
                safe_group_words = [escape_ass_text(w) for w in group_text_raw.split()]
                safe_group_text = format_two_line_ass(safe_group_words)
                f.write(f"Dialogue: 0,{to_ass_timestamp(max(0.0, group_start + subtitle_offset))},{to_ass_timestamp(max(0.0, group_end + subtitle_offset))},Default,,0,0,0,,{safe_group_text}\n")
                continue
            
            # For each word in the group, we create a dialogue line where that word is highlighted
            for idx, active_word in enumerate(group_words):
                line_start = active_word["start"]
                if idx < len(group_words) - 1:
                    line_end = group_words[idx+1]["start"]
                else:
                    line_end = group_end
                
                # Clip times to group boundaries
                line_start = max(line_start, group_start)
                line_end = min(line_end, group_end)
                if line_end <= line_start:
                    continue
                    
                # Apply subtitle offset to shift captions slightly earlier
                offset_start = max(0.0, line_start + subtitle_offset)
                offset_end = max(0.0, line_end + subtitle_offset)
                if offset_end <= offset_start:
                    offset_end = offset_start + 0.1
                    
                # Build highlighted string
                parts = []
                for w in group_words:
                    word_txt = escape_ass_text(w["word"].upper())
                    if w == active_word:
                        parts.append(f"{{\\1c&H00F2FF&}}{word_txt}{{\\1c&HFFFFFF&}}")
                    else:
                        parts.append(word_txt)
                        
                highlighted_text = format_two_line_ass(parts)
                f.write(f"Dialogue: 0,{to_ass_timestamp(offset_start)},{to_ass_timestamp(offset_end)},Default,,0,0,0,,{highlighted_text}\n")
