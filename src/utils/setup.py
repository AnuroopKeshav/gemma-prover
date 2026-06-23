import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def setup():
    load_dotenv(PROJECT_ROOT / ".env")

    os.environ["HF_TOKEN"]

    print("Setup Complete")
