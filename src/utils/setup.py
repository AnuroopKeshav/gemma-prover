import os
import warnings
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

REQUIRED_ENV_VARS = [
    "HF_TOKEN",
    "API_PROVIDER",
    "MODEL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
]


def setup():
    load_dotenv(PROJECT_ROOT / ".env")

    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            warnings.warn(f"{var} not set in environment/.env")

    print("Setup Complete")
