
from pathlib import Path

import pandas as pd

from src.config import DATASET_DIR, validate_dataset_dir


def load_csv(path: Path) -> pd.DataFrame:
    """Load a supplied CSV while preserving identifiers as strings."""
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_practice_questions() -> pd.DataFrame:
    validate_dataset_dir()
    path = DATASET_DIR / "practice" / "questions.csv"
    return load_csv(path)


def load_practice_conversations() -> pd.DataFrame:
    validate_dataset_dir()
    path = DATASET_DIR / "practice" / "conversations.csv"
    return load_csv(path)


def inspect_practice_data() -> None:
    """Print schema and sample rows without modifying the supplied files."""
    questions = load_practice_questions()
    conversations = load_practice_conversations()

    print("\n=== Practice Questions ===")
    print(f"File: {DATASET_DIR / 'practice' / 'questions.csv'}")
    print(f"Rows: {len(questions)}")
    print(f"Columns: {questions.columns.tolist()}")
    print(questions.head(3).to_string(index=False))

    print("\n=== Practice Conversations ===")
    print(f"File: {DATASET_DIR / 'practice' / 'conversations.csv'}")
    print(f"Rows: {len(conversations)}")
    print(f"Columns: {conversations.columns.tolist()}")
    print(conversations.head(3).to_string(index=False))
