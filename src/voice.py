
"""Gnani.ai voice pipeline and batch audio processing for VoiceOps."""

import argparse
import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.config import OUTPUT_DIR
from src.conversation import ConversationController
from src.retrieval import KnowledgeRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOICE_OUTPUT_DIR = Path(OUTPUT_DIR) / "voice"
DEFAULT_LANGUAGE = "en-IN"
DEFAULT_TTS_VOICE = "Kaveri"
DEFAULT_TTS_MODEL = "timbre-v2.5"
MAX_AUDIO_SECONDS = 60
SUPPORTED_INPUT_EXTENSIONS = {".wav", ".mp4"}

load_dotenv(PROJECT_ROOT / ".env")


def _safe_filename(name: str) -> str:
    """Convert an input filename into a safe output basename."""
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "audio"


def _ensure_gnani_key() -> str:
    """Read the required Gnani API key without exposing it in errors."""
    api_key = os.getenv("GNANI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GNANI_API_KEY is not configured. Add it to the project's .env file."
        )
    return api_key


def _make_clients():
    """Create official Gnani STT and TTS clients."""
    _ensure_gnani_key()

    try:
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
    except ImportError as exc:
        raise RuntimeError(
            "Gnani SDK is not installed. Install it with: "
            "pip install gnani-vachana"
        ) from exc

    api_key = os.getenv("GNANI_API_KEY", "").strip()

    try:
        stt_client = GnaniSTTClient(api_key=api_key)
        tts_client = GnaniTTSClient(api_key=api_key)
    except TypeError:
        # Some SDK versions read GNANI_API_KEY directly from the environment.
        stt_client = GnaniSTTClient()
        tts_client = GnaniTTSClient()

    return stt_client, tts_client


def _validate_audio_file(audio_path: Path) -> None:
    """Validate WAV input and enforce the capstone's 60-second clip limit."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        with wave.open(str(audio_path), "rb") as audio:
            duration = audio.getnframes() / float(audio.getframerate())
    except (wave.Error, EOFError, ZeroDivisionError) as exc:
        raise ValueError(f"Unable to read WAV audio: {audio_path.name}") from exc

    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Audio is {duration:.1f} seconds long. "
            f"The capstone limit is {MAX_AUDIO_SECONDS} seconds per clip."
        )

    if duration <= 0:
        raise ValueError("The WAV file contains no audio frames.")


def _prepare_audio_for_stt(audio_path: Path) -> tuple[Path, Path | None]:
    """Return a WAV path for STT and an optional temporary file to clean up."""
    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        _validate_audio_file(audio_path)
        return audio_path, None

    if suffix != ".mp4":
        raise ValueError(
            "Unsupported audio format. Use WAV or MP4 input files."
        )

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "MP4 input requires FFmpeg. Install FFmpeg and add its bin "
            "directory to PATH, then run the command again."
        )

    temporary_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        prefix="voiceops_",
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(audio_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                "-y",
                str(temporary_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip()
            raise ValueError(
                f"Unable to extract audio from MP4 file {audio_path.name}"
                + (f": {detail}" if detail else ".")
            )

        _validate_audio_file(temporary_path)
        return temporary_path, temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def transcribe_audio(
    audio_path: str | Path,
    stt_client=None,
    language_code: str = DEFAULT_LANGUAGE,
) -> str:
    """Transcribe an audio clip using Gnani STT."""
    path = Path(audio_path).expanduser().resolve()
    stt_path, temporary_path = _prepare_audio_for_stt(path)

    try:
        if stt_client is None:
            stt_client, _ = _make_clients()

        result = stt_client.transcribe(
            str(stt_path),
            language_code=language_code,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    if isinstance(result, dict):
        transcript = result.get("transcript") or result.get("text") or ""
    else:
        transcript = str(result or "")

    transcript = str(transcript).strip()
    if not transcript:
        raise RuntimeError(
            f"Gnani STT returned an empty transcript for {path.name}."
        )

    return transcript


def synthesize_audio(
    text: str,
    output_path: str | Path,
    tts_client=None,
    voice: str = DEFAULT_TTS_VOICE,
    model: str = DEFAULT_TTS_MODEL,
    language: str = DEFAULT_LANGUAGE,
) -> Path:
    """Synthesize a response with Gnani TTS and save a WAV file."""
    text = str(text or "").strip()
    if not text:
        raise ValueError("Cannot synthesize an empty response.")

    if tts_client is None:
        _, tts_client = _make_clients()

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    synthesize = tts_client.synthesize
    parameters = inspect.signature(synthesize).parameters

    kwargs = {
        "voice": voice,
        "model": model,
    }

    # The language parameter is supported by timbre-v2.5.
    if "language" in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in parameters.values()
    ):
        kwargs["language"] = language

    # Support the output_file form described by the capstone brief where
    # available, and the current SDK form that returns audio bytes.
    if "output_file" in parameters:
        kwargs["output_file"] = str(destination)
        result = synthesize(text, **kwargs)

        if destination.is_file() and destination.stat().st_size > 0:
            return destination

        if isinstance(result, (bytes, bytearray)):
            destination.write_bytes(result)
            return destination

        raise RuntimeError(
            "Gnani TTS did not create the requested output audio file."
        )

    result = synthesize(text, **kwargs)

    # Some SDK responses return raw audio bytes directly.
    if isinstance(result, (bytes, bytearray)):
        audio_bytes = bytes(result)
    elif isinstance(result, dict):
        audio_bytes = result.get("audio") or result.get("audio_bytes")
        if not isinstance(audio_bytes, (bytes, bytearray)):
            raise RuntimeError(
                "Gnani TTS returned a response without audio bytes."
            )
        audio_bytes = bytes(audio_bytes)
    else:
        raise RuntimeError(
            "Gnani TTS returned an unsupported audio response."
        )

    destination.write_bytes(audio_bytes)
    return destination


def _serialise_evidence_item(item: Any) -> Dict[str, Any]:
    """Convert a retrieved knowledge chunk to JSON-friendly metadata."""
    if isinstance(item, dict):
        data = dict(item)
    elif hasattr(item, "chunk") and hasattr(item, "similarity"):
        chunk = item.chunk
        return {
            "chunk_id": str(chunk.chunk_id),
            "section_id": str(chunk.section_id),
            "source_file": str(chunk.source_file),
            "title": str(chunk.title),
            "category": str(chunk.category),
            "status": str(chunk.status),
            "effective_from": str(chunk.effective_from),
            "similarity": round(float(item.similarity), 4),
            "text": str(chunk.text),
        }
    elif hasattr(item, "__dict__"):
        data = dict(vars(item))
    else:
        data = {"content": str(item)}

    # Avoid writing arbitrarily large source documents into the output JSON.
    result = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[str(key)] = value
        else:
            result[str(key)] = str(value)

    return result


def _retrieve_evidence(
    transcript: str,
    retriever: KnowledgeRetriever,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Capture the evidence retrieved for the customer's transcribed query."""
    try:
        chunks = retriever.retrieve(
            transcript,
            top_k=top_k,
            authoritative_only=True,
        )
    except TypeError:
        # Compatibility with a retriever version that has a smaller signature.
        chunks = retriever.retrieve(transcript, top_k=top_k)

    return [_serialise_evidence_item(item) for item in (chunks or [])]


def process_audio_file(
    audio_path: str | Path,
    output_dir: str | Path = VOICE_OUTPUT_DIR,
    stt_client=None,
    tts_client=None,
    agent: Optional[ConversationController] = None,
    retriever: Optional[KnowledgeRetriever] = None,
    language_code: str = DEFAULT_LANGUAGE,
    voice: str = DEFAULT_TTS_VOICE,
    model: str = DEFAULT_TTS_MODEL,
) -> Dict[str, Any]:
    """
    Process one WAV file and persist transcript, response, evidence and audio.

    STT and TTS use Gnani.ai. The existing conversation controller remains
    responsible for the support response and official mock-tool interactions.
    """
    source = Path(audio_path).expanduser().resolve()
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    basename = _safe_filename(source.name)
    transcript_path = destination_dir / f"{basename}.transcript.txt"
    response_path = destination_dir / f"{basename}.response.txt"
    evidence_path = destination_dir / f"{basename}.evidence.json"
    audio_output_path = destination_dir / f"{basename}.reply.wav"
    result_path = destination_dir / f"{basename}.result.json"

    record: Dict[str, Any] = {
        "source_audio": str(source),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "language_code": language_code,
        "tts_voice": voice,
        "tts_model": model,
        "transcript": "",
        "response": "",
        "retrieved_evidence": [],
        "reply_audio": "",
        "decision": None,
        "tool_calls": [],
        "error": None,
        "status": "started",
    }

    try:
        _validate_audio_file(source)

        if stt_client is None or tts_client is None:
            created_stt, created_tts = _make_clients()
            stt_client = stt_client or created_stt
            tts_client = tts_client or created_tts

        if retriever is None:
            retriever = KnowledgeRetriever()

        if agent is None:
            agent = ConversationController()

        transcript = transcribe_audio(
            source,
            stt_client=stt_client,
            language_code=language_code,
        )
        record["transcript"] = transcript
        transcript_path.write_text(transcript + "\n", encoding="utf-8")

        evidence = _retrieve_evidence(transcript, retriever)
        record["retrieved_evidence"] = evidence
        evidence_path.write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        response = agent.handle_message(transcript)
        record["response"] = response
        record["decision"] = agent.last_decision
        record["tool_calls"] = list(agent.last_tool_calls)
        response_path.write_text(response + "\n", encoding="utf-8")

        reply_audio = synthesize_audio(
            response,
            audio_output_path,
            tts_client=tts_client,
            voice=voice,
            model=model,
            language=language_code,
        )

        record["reply_audio"] = str(reply_audio)
        record["status"] = "success"

    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"

    result_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    record["result_json"] = str(result_path)
    return record


def process_audio_folder(
    input_dir: str | Path,
    output_dir: str | Path = VOICE_OUTPUT_DIR,
    language_code: str = DEFAULT_LANGUAGE,
    voice: str = DEFAULT_TTS_VOICE,
    model: str = DEFAULT_TTS_MODEL,
) -> Dict[str, Any]:
    """
    Process every WAV file in a folder and write a batch manifest.

    A fresh conversation controller is created for each audio file because
    the folder may contain unrelated callers.
    """
    source_dir = Path(input_dir).expanduser().resolve()
    destination_dir = Path(output_dir).expanduser().resolve()

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Audio input folder not found: {source_dir}")

    audio_files = sorted(
        path for path in source_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
    )

    if not audio_files:
        raise FileNotFoundError(
            f"No WAV or MP4 files were found in {source_dir}."
        )

    destination_dir.mkdir(parents=True, exist_ok=True)

    # Create clients once per batch, not once per clip.
    stt_client, tts_client = _make_clients()
    retriever = KnowledgeRetriever()

    results = []

    for index, audio_path in enumerate(audio_files, start=1):
        print(f"[{index}/{len(audio_files)}] Processing {audio_path.name}")

        result = process_audio_file(
            audio_path=audio_path,
            output_dir=destination_dir,
            stt_client=stt_client,
            tts_client=tts_client,
            agent=ConversationController(),
            retriever=retriever,
            language_code=language_code,
            voice=voice,
            model=model,
        )
        results.append(result)

        print(f"  Status: {result['status']}")
        if result.get("error"):
            print(f"  Error: {result['error']}")

    manifest_path = destination_dir / "batch_manifest.json"
    summary = {
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(source_dir),
        "output_directory": str(destination_dir),
        "files_found": len(audio_files),
        "successful": sum(item["status"] == "success" for item in results),
        "failed": sum(item["status"] != "success" for item in results),
        "results": results,
    }

    manifest_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nBatch processing complete.")
    print(f"Manifest: {manifest_path}")
    print(
        f"Successful: {summary['successful']} | "
        f"Failed: {summary['failed']}"
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ZentroMart Gnani.ai voice pipeline."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one WAV file or a folder containing WAV files.",
    )
    parser.add_argument(
        "--output",
        default=str(VOICE_OUTPUT_DIR),
        help="Directory for transcripts, responses, evidence and generated audio.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Gnani language code, for example en-IN or hi-IN.",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_TTS_VOICE,
        help="Gnani TTS voice.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TTS_MODEL,
        help="Gnani TTS model.",
    )

    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()

    if input_path.is_dir():
        process_audio_folder(
            input_dir=input_path,
            output_dir=args.output,
            language_code=args.language,
            voice=args.voice,
            model=args.model,
        )
    else:
        result = process_audio_file(
            audio_path=input_path,
            output_dir=args.output,
            language_code=args.language,
            voice=args.voice,
            model=args.model,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
