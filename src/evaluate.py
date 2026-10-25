
import argparse
import json
from datetime import datetime
from pathlib import Path
import re
import ast
from datetime import datetime, timezone
from src.config import OUTPUT_DIR
from src.rag import answer_with_rag
from src.retrieval import KnowledgeRetriever
import pandas as pd
from tqdm import tqdm
import src.conversation as conversation_module
from src.conversation import ConversationController
from collections import Counter

from src.config import DATASET_DIR
from src.dataset import load_csv
from src.config import BASELINE_OUTPUT_DIR
from src.dataset import (
    load_practice_conversations,
    load_practice_questions,
)
from src.baseline import answer_baseline


QUESTION_COLUMNS = (
    "asr_text",
    "question",
    "user_question",
    "text",
)


def find_question_column(df: pd.DataFrame) -> str:
    """Identify the question text column from the actual CSV schema."""
    for column in QUESTION_COLUMNS:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not identify the question text column. "
        f"Available columns: {df.columns.tolist()}"
    )


def evaluate_baseline(
    limit: int | None = None,
    pause_on_error: bool = False,
) -> Path:
    questions = load_practice_questions()

    if questions.empty:
        raise ValueError("The practice questions file contains no rows.")

    question_column = find_question_column(questions)

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be greater than zero.")
        questions = questions.head(limit).copy()

    results = []

    for row_index, row in tqdm(
        questions.iterrows(),
        total=len(questions),
        desc="Prompt-only baseline",
    ):
        question = str(row[question_column]).strip()

        result = row.to_dict()
        result["baseline_question"] = question
        result["baseline_answer"] = ""
        result["baseline_error"] = ""

        if not question:
            result["baseline_error"] = "Empty question"
            results.append(result)
            continue

        try:
            result["baseline_answer"] = answer_baseline(question)
        except Exception as exc:
            result["baseline_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            if pause_on_error:
                raise

        results.append(result)

    output_path = (
        BASELINE_OUTPUT_DIR / "baseline_results.csv"
    )
    pd.DataFrame(results).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "stage": "step_1_prompt_only_baseline",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(DATASET_DIR),
        "questions_attempted": len(results),
        "successful_answers": sum(
            bool(item["baseline_answer"])
            for item in results
        ),
        "errors": sum(
            bool(item["baseline_error"])
            for item in results
        ),
        "question_column": question_column,
        "results_file": str(output_path),
    }

    summary_path = BASELINE_OUTPUT_DIR / "baseline_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nBaseline evaluation complete.")
    print(f"Results: {output_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return output_path


def inspect_conversations() -> None:
    conversations = load_practice_conversations()

    print("\n=== Conversation Dataset ===")
    print(f"Rows: {len(conversations)}")
    print(f"Columns: {conversations.columns.tolist()}")
    print(conversations.head(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the ZentroMart prompt-only baseline."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N practice questions.",
    )
    parser.add_argument(
        "--inspect-conversations",
        action="store_true",
        help="Print the conversation dataset schema and sample.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if an LLM request fails.",
    )

    args = parser.parse_args()

    if args.inspect_conversations:
        inspect_conversations()
    else:
        evaluate_baseline(
            limit=args.limit,
            pause_on_error=args.stop_on_error,
        )


def _split_expected_terms(value):
    """
    Each slash-separated group represents acceptable alternatives.
    Example: 'instant / 2 hours / two hours' is one requirement.
    """
    if value is None:
        return []

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []

    return [
        [alternative.strip().lower() for alternative in re.split(r"\s*/\s*", group)
         if alternative.strip()]
        for group in re.split(r"\s*;\s*|\n", text)
        if group.strip()
    ]


def _split_ids(value):
    if value is None:
        return []

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []

    return [
        item.strip()
        for item in re.split(r"\s*[,;|]\s*", text)
        if item.strip()
    ]


def _is_false(value):
    return str(value).strip().lower() in {"false", "0", "no", "n"}


def run_rag_evaluation():
    questions = load_practice_questions()
    if questions.empty:
        raise ValueError("practice/questions.csv is empty.")

    required_columns = {
        "id", "question", "answerable", "gold_evidence",
        "must_include", "must_not_include",
    }
    missing = required_columns - set(questions.columns)
    if missing:
        raise ValueError(
            f"Practice questions file is missing columns: {sorted(missing)}"
        )

    retriever = KnowledgeRetriever()
    rows = []

    total = len(questions)

    for index, row in questions.iterrows():
        question_id = str(row.get("id", f"Q{index + 1}"))
        question = str(row.get("question", "")).strip()

        if not question:
            continue

        result = answer_with_rag(question, retriever=retriever)
        answer = result["answer"]
        answer_lower = answer.lower()

        must_include_groups = _split_expected_terms(row.get("must_include", ""))
        must_not_include_groups = _split_expected_terms(row.get("must_not_include", ""))

        required_passes = [
            any(term in answer_lower for term in group)
            for group in must_include_groups
        ]
        prohibited_passes = [
            not any(term in answer_lower for term in group)
            for group in must_not_include_groups
        ]

        retrieved_ids = {
            str(item.get("section_id", "")).strip()
            for item in result["evidence"]
            if item.get("section_id")
        }
        gold_ids = set(_split_ids(row.get("gold_evidence", "")))

        # A section is considered retrieved if its ID appears in the
        # retrieved evidence list, even if its similarity is 0 after
        # deliberate broad-policy context expansion.
        evidence_hits = sorted(retrieved_ids & gold_ids)
        evidence_recall = (
            len(evidence_hits) / len(gold_ids) if gold_ids else None
        )

        answerable = not _is_false(row.get("answerable", ""))
        uncertainty_response = not result["grounded"]

        rows.append({
            "id": question_id,
            "question": question,
            "intent": row.get("intent", ""),
            "answerable": answerable,
            "expected_decision": row.get("expected_decision", ""),
            "grounded": result["grounded"],
            "uncertainty_response": uncertainty_response,
            "expected_uncertainty": not answerable,
            "uncertainty_correct": (
                uncertainty_response if not answerable
                else not uncertainty_response
            ),
            "required_groups_total": len(required_passes),
            "required_groups_passed": sum(required_passes),
            "must_include_pass": (
                all(required_passes) if required_passes else None
            ),
            "prohibited_groups_total": len(prohibited_passes),
            "prohibited_groups_passed": sum(prohibited_passes),
            "must_not_include_pass": (
                all(prohibited_passes) if prohibited_passes else None
            ),
            "gold_evidence": ";".join(sorted(gold_ids)),
            "retrieved_evidence": ";".join(sorted(retrieved_ids)),
            "evidence_hits": ";".join(evidence_hits),
            "gold_evidence_recall": evidence_recall,
            "answer": answer,
            "evidence_details": json.dumps(
                result["evidence"], ensure_ascii=False
            ),
        })

        print(f"[{len(rows)}/{total}] Evaluated {question_id}")

    results = pd.DataFrame(rows)

    output_dir = OUTPUT_DIR / "rag"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "rag_evaluation.csv"
    json_path = output_dir / "rag_evaluation_summary.json"

    results.to_csv(csv_path, index=False, encoding="utf-8-sig")

    def rate(column):
        values = results[column].dropna()
        if values.empty:
            return None
        return round(float(values.mean()), 4)

    answerable_rows = results[results["answerable"] == True]
    unanswerable_rows = results[results["answerable"] == False]

    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions_evaluated": len(results),
        "answerable_questions": len(answerable_rows),
        "unanswerable_questions": len(unanswerable_rows),
        "must_include_pass_rate": rate("must_include_pass"),
        "must_not_include_pass_rate": rate("must_not_include_pass"),
        "gold_evidence_recall_mean": rate("gold_evidence_recall"),
        "uncertainty_correct_rate_all": rate("uncertainty_correct"),
        "answerable_grounded_rate": (
            round(float(answerable_rows["grounded"].mean()), 4)
            if not answerable_rows.empty else None
        ),
        "unanswerable_correct_uncertainty_rate": (
            round(float(unanswerable_rows["uncertainty_correct"].mean()), 4)
            if not unanswerable_rows.empty else None
        ),
        "outputs": {
            "csv": str(csv_path),
            "summary_json": str(json_path),
        },
        "metric_notes": [
            "Must-include checks use dataset-provided alternative terms.",
            "Gold evidence recall measures section-ID overlap, not answer correctness.",
            "Uncertainty correctness compares the grounded flag with the answerable field.",
            "Human review is still required to assess factual support and citation accuracy.",
        ],
    }

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("RAG EVALUATION SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDetailed results: {csv_path}")
    print(f"Summary: {json_path}")

    return summary

def run_conversation_evaluation():
    """Evaluate the agent against the supplied multi-turn practice dataset."""

    conversations_path = DATASET_DIR / "practice" / "conversations.csv"

    if not conversations_path.is_file():
        raise FileNotFoundError(
            f"Practice conversations file not found: {conversations_path}"
        )

    df = load_csv(conversations_path)

    required_columns = {
        "conversation_id",
        "turn",
        "customer_message",
        "expected_decision",
        "expected_tool_calls",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Conversation dataset is missing columns: {sorted(missing)}"
        )

    results = []
    counters = Counter()

    # Each conversation starts with a fresh controller.
    for conversation_id, group in df.groupby(
        "conversation_id", sort=False
    ):
        agent = ConversationController()

        group = group.copy()
        group["_turn_number"] = group["turn"].astype(int)
        group = group.sort_values("_turn_number")

        for _, row in group.iterrows():
            message = str(row["customer_message"]).strip()

            if not message:
                continue

            expected_decision = str(row["expected_decision"]).strip()
            expected_tool_calls = str(row["expected_tool_calls"]).strip()

            error = None
            reply = ""

            try:
                reply = agent.handle_message(message)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            # Capture actual tool-result information for auditing.
            tool_result = agent.last_tool_result
            tool_result_json = (
                json.dumps(tool_result, ensure_ascii=False, default=str)
                if tool_result is not None
                else ""
            )

            # These are recorded for inspection, not treated as proof
            # that the expected decision or tool call was correct.
            results.append(
                {
                    "conversation_id": str(conversation_id),
                    "turn": str(row["turn"]),
                    "customer_message": message,
                    "expected_decision": expected_decision,
                    "expected_tool_calls": expected_tool_calls,
                    "assistant_reply": reply,
                    "tool_result": tool_result_json,
                    "error": error or "",
                }
            )

            counters["turns"] += 1
            if error:
                counters["errors"] += 1
            if tool_result is not None:
                counters["turns_with_tool_results"] += 1

    output_dir = OUTPUT_DIR / "conversations"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "conversation_evaluation.jsonl"
    with results_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conversation_count": int(df["conversation_id"].nunique()),
        "turns_evaluated": counters["turns"],
        "turns_with_errors": counters["errors"],
        "turns_with_tool_results": counters["turns_with_tool_results"],
        "results_file": str(results_path.resolve()),
        "note": (
            "This initial evaluation records outputs and tool results. "
            "Expected decisions and tool-call correctness require explicit "
            "normalization before pass rates can be reported."
        ),
    }

    summary_path = output_dir / "conversation_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _parse_expected_tool_call(value):
    """Parse expected tool calls from the practice CSV."""
    value = str(value or "").strip()

    if not value:
        return None

    match = re.fullmatch(
        r"\s*(lookup_order|create_ticket)\s*\((.*)\)\s*",
        value,
    )
    if not match:
        return {"tool": "__unparsed__", "arguments": {}}

    tool_name, raw_arguments = match.groups()
    arguments = {}

    if not raw_arguments.strip():
        return {"tool": tool_name, "arguments": arguments}

    pattern = re.compile(
        r"\s*(order_id|verification_value|summary|reason|intent)"
        r"\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([0-9]+))\s*(?:,|$)"
    )

    position = 0
    while position < len(raw_arguments):
        item = pattern.match(raw_arguments, position)

        if not item:
            return {"tool": "__unparsed__", "arguments": {}}

        key, single_quoted, double_quoted, digits = item.groups()

        parsed_value = (
            single_quoted
            if single_quoted is not None
            else double_quoted
            if double_quoted is not None
            else digits
        )

        arguments[key] = str(parsed_value)
        position = item.end()

    return {"tool": tool_name, "arguments": arguments}


def _conversation_tool_match(expected, actual_calls):
    """Compare tool names and only the arguments specified by the dataset."""
    if expected is None:
        return len(actual_calls) == 0

    if expected["tool"] == "__unparsed__":
        return False

    for actual in actual_calls:
        if actual.get("tool") != expected["tool"]:
            continue

        actual_arguments = actual.get("arguments", {})

        if all(
            key in actual_arguments
            and str(actual_arguments[key]) == str(expected_value)
            for key, expected_value in expected["arguments"].items()
        ):
            return True

    return False


def run_conversation_evaluation_v2():
    """Evaluate conversation decisions, tool calls, and runtime errors."""
    dataset = load_practice_conversations()

    output_dir = Path(OUTPUT_DIR) / "conversations"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "conversation_evaluation_v2.jsonl"
    summary_path = output_dir / "conversation_evaluation_v2_summary.json"

    records = []
    runtime_errors = 0
    tool_comparable = 0
    tool_matches = 0
    decision_comparable = 0
    decision_matches = 0
    actual_tool_counts = Counter()

    valid_decisions = {"clarify", "answer", "escalate", "refuse"}

    for conversation_id, group in dataset.groupby(
        "conversation_id", sort=False
    ):
        agent = ConversationController()

        for _, row in group.sort_values("turn").iterrows():
            message = str(row.get("customer_message", ""))
            runtime_error = None
            reply = None

            try:
                reply = agent.handle_message(message)
            except Exception as exc:
                runtime_errors += 1
                runtime_error = f"{type(exc).__name__}: {exc}"

            actual_calls = list(getattr(agent, "last_tool_calls", []) or [])
            expected_raw = str(row.get("expected_tool_calls", "") or "")
            expected_tool = _parse_expected_tool_call(expected_raw)

            tool_match = _conversation_tool_match(
                expected_tool,
                actual_calls,
            )

            tool_comparable += 1
            tool_matches += int(tool_match)

            for call in actual_calls:
                actual_tool_counts[call.get("tool", "unknown")] += 1

            expected_decision = str(
                row.get("expected_decision", "")
            ).strip().lower()

            actual_decision = getattr(agent, "last_decision", None)

            if actual_decision is not None:
                actual_decision = str(actual_decision).strip().lower()

            decision_match = None

            if (
                expected_decision in valid_decisions
                and actual_decision in valid_decisions
            ):
                decision_match = actual_decision == expected_decision
                decision_comparable += 1
                decision_matches += int(decision_match)

            records.append({
                "conversation_id": str(conversation_id),
                "turn": str(row.get("turn", "")),
                "scenario": str(row.get("scenario", "")),
                "customer_message": message,
                "reply": reply,
                "expected_decision": expected_decision,
                "actual_decision": actual_decision,
                "decision_match": decision_match,
                "expected_tool_calls": expected_raw,
                "actual_tool_calls": actual_calls,
                "tool_call_match": tool_match,
                "runtime_error": runtime_error,
            })

    with results_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conversations": int(dataset["conversation_id"].nunique()),
        "turns": len(records),
        "runtime_errors": runtime_errors,
        "tool_call_comparable_turns": tool_comparable,
        "tool_call_matches": tool_matches,
        "tool_call_agreement_rate": (
            round(tool_matches / tool_comparable, 4)
            if tool_comparable else None
        ),
        "decision_comparable_turns": decision_comparable,
        "decision_matches": decision_matches,
        "decision_agreement_rate": (
            round(decision_matches / decision_comparable, 4)
            if decision_comparable else None
        ),
        "actual_tool_counts": dict(actual_tool_counts),
        "results_file": str(results_path.resolve()),
        "summary_file": str(summary_path.resolve()),
        "decision_agreement_note": (
            "Decisions are compared only when both labels match the "
            "practice dataset vocabulary."
        ),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
