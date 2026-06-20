import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    print("Warning: GEMINI_API_KEY is not set or is using the default placeholder.")

if not REPLICATE_API_TOKEN or REPLICATE_API_TOKEN == "your_replicate_api_token_here":
    print("Warning: REPLICATE_API_TOKEN is not set or is using the default placeholder.")
