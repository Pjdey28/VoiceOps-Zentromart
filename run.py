
import argparse

from src.config import validate_dataset_dir, validate_llm_config
from src.dataset import inspect_practice_data
from src.evaluate import evaluate_baseline, inspect_conversations
from src.evaluate import run_rag_evaluation
from src.evaluate import run_conversation_evaluation
from src.evaluate import run_conversation_evaluation_v2
from src.voice import process_audio_folder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ZentroMart VoiceOps capstone runner."
    )
    parser.add_argument(
        "command",
        choices=(
            "check-data",
            "inspect",
            "baseline",
            "conversations",
            "rag-evaluate",
            "conversation-evaluate",
            "conversation-evaluate-v2",
            "voice-batch",
        ),
        help=(
            "check-data: validate dataset location; "
            "inspect: display practice schemas; "
            "baseline: run prompt-only evaluation; "
            "conversations: inspect conversation cases; "
            "rag-evaluate: run RAG evaluation; "
            "conversation-evaluate: run conversation evaluation; "
            "conversation-evaluate-v2: run conversation evaluation (v2); "
            "voice-batch: process a folder of WAV files with Gnani STT/TTS."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum practice questions to run in baseline mode.",
    )
    parser.add_argument(
        "--input",
        help="Input folder for the voice-batch command.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder for the voice-batch command.",
    )
    args = parser.parse_args()

    if args.command == "check-data":
        validate_dataset_dir()
        print(f"Dataset directory found: ")
        print("Run the dataset's own validator next.")
    elif args.command == "inspect":
        validate_dataset_dir()
        inspect_practice_data()
    elif args.command == "baseline":
        validate_dataset_dir()
        validate_llm_config()
        evaluate_baseline(limit=args.limit)
    elif args.command == "conversations":
        validate_dataset_dir()
        inspect_conversations()
    elif args.command == "rag-evaluate":
        run_rag_evaluation()
    elif args.command == "conversation-evaluate":
        run_conversation_evaluation()
    elif args.command == "conversation-evaluate-v2":
        run_conversation_evaluation_v2()
    elif args.command == "voice-batch":
        if not args.input:
            parser.error("voice-batch requires --input <folder>")
        kwargs = {}
        if args.output:
            kwargs["output_dir"] = args.output
        process_audio_folder(input_dir=args.input, **kwargs)

if __name__ == "__main__":
    main()
