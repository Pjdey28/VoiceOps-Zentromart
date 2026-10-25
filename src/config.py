
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _configure_console_encoding() -> None:
    """Keep Unicode dataset content printable on Windows consoles."""
    if os.name != "nt":
        return

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


_configure_console_encoding()

# Project root is the parent of the src directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2", "").strip()
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
).strip()

_dataset_dir = os.getenv(
    "DATASET_DIR",
    "../Capstone_Project_Dataset_FTTT_2026",
).strip()

DATASET_DIR = Path(_dataset_dir).expanduser()

# Relative dataset paths are resolved from the project root.
if not DATASET_DIR.is_absolute():
    DATASET_DIR = (PROJECT_ROOT / DATASET_DIR).resolve()

OUTPUT_DIR = PROJECT_ROOT / "outputs"
BASELINE_OUTPUT_DIR = OUTPUT_DIR / "baseline"
RAG_OUTPUT_DIR = OUTPUT_DIR / "rag"
VOICE_OUTPUT_DIR = OUTPUT_DIR / "voice"

for directory in (
    BASELINE_OUTPUT_DIR,
    RAG_OUTPUT_DIR,
    VOICE_OUTPUT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


def validate_llm_config() -> None:
    """Fail early when the LLM environment configuration is missing."""
    if not GROQ_API_KEY and not GROQ_API_KEY_2:
        raise RuntimeError(
            "At least one Groq API key is required. Add GROQ_API_KEY or "
            "GROQ_API_KEY_2 to the project's .env file."
        )

    if not GROQ_MODEL:
        raise RuntimeError(
            "GROQ_MODEL is empty. Set it in the .env file."
        )


def validate_dataset_dir() -> None:
    """Check that the configured dataset directory exists."""
    if not DATASET_DIR.exists() or not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}\n"
            "Update DATASET_DIR in .env to point to the extracted dataset."
        )
