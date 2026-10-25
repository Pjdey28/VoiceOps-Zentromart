
from src.llm import generate_response
from src.prompts import build_baseline_messages


def answer_baseline(question: str) -> str:
    """
    Answer one question using the prompt-only baseline.

    This stage deliberately does not retrieve knowledge-base documents
    or access customer/order records.
    """
    messages = build_baseline_messages(question)
    return generate_response(messages)
