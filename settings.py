"""Load variables from `.env` in the project root (python-dotenv)."""

from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")
