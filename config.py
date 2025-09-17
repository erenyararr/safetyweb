import os
from dotenv import load_dotenv

# Load variables from a local .env file if present
load_dotenv()

# Read the OpenAI API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY", "")

if not API_KEY:
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Set it in your environment or in a .env file."
    )
