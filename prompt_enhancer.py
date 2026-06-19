import json
import google.generativeai as genai
from config import GEMINI_API_KEY

def setup_gemini():
    """Configures the Gemini API client."""
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        raise ValueError("Valid Gemini API key not found. Please check your .env file.")

def generate_story_scenes(topic, num_clips):
    """
    Uses Gemini to expand a simple topic into a sequence of 'num_clips' detailed scenes.
    Returns a list of strings (the prompts).
    """
    setup_gemini()
    
    # We use gemini-2.5-flash for speed and intelligence
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    system_instruction = (
        f"You are an expert Storyboard Director for a CARTOON video. "
        f"The user wants a story about their topic, broken down exactly into {num_clips} sequential scenes. "
        "For each scene, write a highly detailed, vivid, and descriptive prompt for a Text-to-Video AI. "
        "Ensure every prompt explicitly mentions styles like '2D cartoon animation', "
        "'vibrant colors', 'smooth animation', 'Studio Ghibli style', 'high quality'. "
        "Keep each scene prompt under 60 words. "
        "Return the output STRICTLY as a valid JSON array of strings. Do not include any other markdown or text."
    )
    
    full_prompt = f"{system_instruction}\n\nTopic: {topic}"
    
    try:
        response = model.generate_content(full_prompt)
        text_output = response.text.strip()
        
        # Clean up in case Gemini wraps it in ```json ... ```
        if text_output.startswith("```json"):
            text_output = text_output[7:-3].strip()
        elif text_output.startswith("```"):
            text_output = text_output[3:-3].strip()
            
        scenes = json.loads(text_output)
        
        if not isinstance(scenes, list):
            raise ValueError("Output is not a JSON list.")
            
        # Ensure we return exactly the requested number of clips (truncate or pad)
        if len(scenes) > num_clips:
            scenes = scenes[:num_clips]
        
        return scenes
    except Exception as e:
        print(f"Error communicating with Gemini API or parsing JSON: {e}")
        # Fallback list if it fails
        fallback_prompt = f"{topic}, 2D cartoon animation style, high quality, vibrant colors"
        return [fallback_prompt] * num_clips

if __name__ == "__main__":
    # Test the enhancer
    test_topic = "a heroic mouse"
    try:
        print(f"Original Topic: {test_topic}")
        scenes = generate_story_scenes(test_topic, 3)
        for i, scene in enumerate(scenes):
            print(f"Scene {i+1}: {scene}")
    except Exception as e:
        print(e)
