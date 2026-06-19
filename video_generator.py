import os
import requests
import replicate
from config import REPLICATE_API_TOKEN

# Note: The replicate package automatically picks up REPLICATE_API_TOKEN from the environment,
# but we ensure it's set here just in case.
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

def generate_video(prompt):
    """
    Sends the prompt to Replicate API using the official SDK.
    """
    if not REPLICATE_API_TOKEN or REPLICATE_API_TOKEN == "your_replicate_api_token_here":
        raise ValueError("Valid Replicate API token not found. Please check your .env file.")

    print("Sending request to Replicate Video Generation API (via official SDK)...")
    
    try:
        # Using Hotshot-XL for fast generation
        model_id = "lucataco/hotshot-xl:8f5e1cd12b26dc832c3216834b6b60e340a5a3a0e4ce6cefbdf142b936d8f635"
        
        # replicate.run automatically handles polling and returns the final output
        output = replicate.run(
            model_id,
            input={
                "prompt": prompt,
                "mp4": True,
                "steps": 30
            }
        )
        
        print("Video generation successful!")
        
        # The output might be a single string (URL) or a list of strings
        if isinstance(output, list):
            return output[0]
        return output
        
    except Exception as e:
        raise Exception(f"Replicate API Error: {e}")

def download_video(video_url, output_filename="output.mp4"):
    """Downloads the generated video from the URL."""
    print(f"Downloading video from Replicate...")
    response = requests.get(video_url)
    
    if response.status_code == 200:
        with open(output_filename, 'wb') as f:
            f.write(response.content)
        print(f"Video saved as {output_filename}")
        return os.path.abspath(output_filename)
    else:
        raise Exception(f"Failed to download video: {response.status_code}")

if __name__ == "__main__":
    pass
