"""Interactive Streamlit chat for the ZentroMart VoiceOps assistant."""

import os
import tempfile
from pathlib import Path
from typing import Any

# Transformers exposes optional vision modules through lazy imports. Streamlit's
# source watcher probes those modules even though this app only handles audio.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st

from src.conversation import ConversationController
from src.retrieval import KnowledgeRetriever
from src.voice import (
    DEFAULT_LANGUAGE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    _make_clients,
    _retrieve_evidence,
    synthesize_audio,
    transcribe_audio,
)


st.set_page_config(page_title="ZentroMart VoiceOps", page_icon="🛒")
st.title("ZentroMart Support Assistant")
st.caption("Ask by text or upload a WAV/MP4 customer message. The assistant remembers follow-ups until you reset the conversation.")


def _initialise_session() -> None:
    if "agent" not in st.session_state:
        st.session_state.agent = ConversationController()
        st.session_state.retriever = KnowledgeRetriever()
        st.session_state.turns = []


def _format_evidence(evidence: list[dict[str, Any]]) -> None:
    if not evidence:
        st.info("No document evidence was retrieved for this turn.")
        return

    for index, item in enumerate(evidence, start=1):
        title = item.get("title") or item.get("section_id") or "Knowledge-base section"
        source = item.get("source_file", "Unknown source")
        similarity = item.get("similarity")
        label = f"{index}. {title} — {source}"
        if similarity is not None:
            label += f" (match {similarity:.3f})"
        with st.expander(label):
            st.markdown(str(item.get("text", "")).strip())
            section_id = item.get("section_id")
            if section_id:
                st.caption(f"Section: {section_id}")


def _render_turn(turn: dict[str, Any], index: int) -> None:
    with st.chat_message("user"):
        st.markdown(f"**Customer message {index}:**")
        st.write(turn["transcript"])
        if turn.get("input_audio"):
            st.audio(turn["input_audio"])

    with st.chat_message("assistant"):
        st.markdown("**ZentroMart assistant:**")
        st.write(turn["response"])
        if turn.get("reply_audio"):
            st.audio(turn["reply_audio"], format="audio/wav")

        with st.expander("Document evidence used"):
            _format_evidence(turn.get("evidence", []))

        tool_calls = turn.get("tool_calls", [])
        if tool_calls:
            with st.expander("Verified support actions"):
                for call in tool_calls:
                    name = call.get("name") or call.get("tool") or "Support tool"
                    result = call.get("result")
                    st.markdown(f"**{name}**")
                    if result:
                        st.write(str(result))


def _synthesise_reply(response: str) -> str | None:
    """Create reply audio for the UI without hiding a successful text reply."""
    try:
        _, tts_client = _make_clients()
        output = Path(tempfile.gettempdir()) / "zentromart_voiceops_reply.wav"
        reply_path = synthesize_audio(
            response,
            output,
            tts_client=tts_client,
            voice=DEFAULT_TTS_VOICE,
            model=DEFAULT_TTS_MODEL,
            language=DEFAULT_LANGUAGE,
        )
        # Use a unique file in the session output so Streamlit can play it
        # after subsequent reruns.
        saved_path = Path("sample_outputs") / "conversation" / f"reply_{len(st.session_state.turns) + 1}.wav"
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path.write_bytes(reply_path.read_bytes())
        return str(saved_path.resolve())
    except Exception as exc:
        st.warning(f"Reply audio could not be generated: {type(exc).__name__}: {exc}")
        return None


def _handle_text_turn(transcript: str, input_audio: str | None = None) -> None:
    transcript = transcript.strip()
    if not transcript:
        st.warning("Please enter a message or provide an audio file containing speech.")
        return

    with st.spinner("Preparing the response..."):
        evidence = _retrieve_evidence(transcript, st.session_state.retriever)
        response = st.session_state.agent.handle_message(transcript)
        reply_audio = _synthesise_reply(response)

    st.session_state.turns.append(
        {
            "transcript": transcript,
            "response": response,
            "evidence": evidence,
            "reply_audio": reply_audio,
            "input_audio": input_audio,
            "tool_calls": list(st.session_state.agent.last_tool_calls),
        }
    )


def _handle_audio_turn(uploaded_file: Any) -> None:
    suffix = Path(uploaded_file.name).suffix.lower()
    input_path = (
        Path("sample_outputs")
        / "conversation"
        / f"input_{len(st.session_state.turns) + 1}{suffix}"
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(uploaded_file.getvalue())

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
        temporary.write(input_path.read_bytes())
        temporary_path = Path(temporary.name)

    try:
        with st.spinner("Transcribing and preparing the response..."):
            stt_client, _ = _make_clients()
            transcript = transcribe_audio(
                temporary_path,
                stt_client=stt_client,
                language_code=DEFAULT_LANGUAGE,
            )
        _handle_text_turn(transcript, input_audio=str(input_path.resolve()))
    finally:
        temporary_path.unlink(missing_ok=True)


_initialise_session()

if st.button("Reset conversation", type="secondary"):
    st.session_state.agent.reset()
    st.session_state.turns = []
    st.rerun()

for turn_index, turn in enumerate(st.session_state.turns, start=1):
    _render_turn(turn, turn_index)

st.divider()
st.subheader("Send a message")
text_message = st.chat_input("Type your customer message here")
if text_message:
    try:
        _handle_text_turn(text_message)
        st.rerun()
    except Exception as exc:
        st.error(f"Unable to process the message: {type(exc).__name__}: {exc}")

uploaded_file = st.file_uploader(
    "Or upload a customer audio message",
    type=["wav", "mp4"],
    help="MP4 audio is decoded locally before being sent to Gnani STT.",
)
if uploaded_file and st.button("Send audio message"):
    try:
        _handle_audio_turn(uploaded_file)
        st.rerun()
    except Exception as exc:
        st.error(f"Unable to process the audio: {type(exc).__name__}: {exc}")
