
"""Inspect the official practice labels before implementing evaluation metrics."""

from src.config import DATASET_DIR
from src.dataset import load_csv


def main():
    path = DATASET_DIR / "practice" / "conversations.csv"

    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = load_csv(path)

    print("\n=== Expected decisions ===")
    print(df["expected_decision"].value_counts(dropna=False).to_string())

    print("\n=== Expected tool-call values ===")
    print(df["expected_tool_calls"].value_counts(dropna=False).to_string())

    print("\n=== Representative conversation turns ===")
    columns = [
        "conversation_id",
        "turn",
        "scenario",
        "customer_message",
        "expected_decision",
        "expected_tool_calls",
        "notes",
    ]
    columns = [column for column in columns if column in df.columns]

    print(
        df[columns]
        .groupby("conversation_id", sort=False)
        .head(3)
        .head(18)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
