
"""Inspect conversation evaluation mismatches without rerunning the agent."""

import json
from pathlib import Path

from src.config import OUTPUT_DIR


def main():
    result_path = (
        Path(OUTPUT_DIR)
        / "conversations"
        / "conversation_evaluation_v2.jsonl"
    )

    if not result_path.is_file():
        raise FileNotFoundError(
            f"Evaluation results not found: {result_path}\n"
            "Run: python run.py conversation-evaluate-v2"
        )

    records = []
    with result_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    tool_mismatches = [
        record for record in records
        if not record.get("tool_call_match", False)
    ]
    decision_mismatches = [
        record for record in records
        if record.get("decision_match") is False
    ]

    print(f"Total turns: {len(records)}")
    print(f"Tool-call mismatches: {len(tool_mismatches)}")
    print(f"Decision mismatches: {len(decision_mismatches)}")

    print("\n" + "=" * 80)
    print("TOOL-CALL MISMATCHES")
    print("=" * 80)

    for record in tool_mismatches:
        print(f"\nConversation: {record.get('conversation_id')}")
        print(f"Turn: {record.get('turn')}")
        print(f"Scenario: {record.get('scenario')}")
        print(f"Customer: {record.get('customer_message')}")
        print(f"Expected decision: {record.get('expected_decision')}")
        print(f"Actual decision: {record.get('actual_decision')}")
        print(f"Expected tool: {record.get('expected_tool_calls')}")
        print(
            "Actual tools: "
            + json.dumps(
                record.get("actual_tool_calls", []),
                ensure_ascii=False,
            )
        )
        if record.get("runtime_error"):
            print(f"Runtime error: {record['runtime_error']}")

    print("\n" + "=" * 80)
    print("DECISION MISMATCHES")
    print("=" * 80)

    for record in decision_mismatches:
        print(
            f"\nConversation {record.get('conversation_id')}, "
            f"turn {record.get('turn')}"
        )
        print(f"Customer: {record.get('customer_message')}")
        print(f"Expected: {record.get('expected_decision')}")
        print(f"Actual: {record.get('actual_decision')}")
        print(f"Reply: {record.get('reply')}")

    print("\nFull results remain unchanged:")
    print(result_path.resolve())


if __name__ == "__main__":
    main()
