import os
import tempfile
from pathlib import Path

# Transformers exposes optional vision modules through lazy imports. Streamlit's
# source watcher probes those modules even though this app only handles audio.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
from src.voice import process_audio_file
from src.conversation import ConversationController
from src.retrieval import KnowledgeRetriever

st.set_page_config(page_title="ZentroMart VoiceOps")
st.title("ZentroMart VoiceOps Demo")

# Maintain the agent in session state so it remembers follow-ups
if "agent" not in st.session_state:
    st.session_state.agent = ConversationController()
    st.session_state.retriever = KnowledgeRetriever()

if st.button("Reset Conversation"):
    st.session_state.agent.reset()
    st.success("Conversation history cleared.")

uploaded_file = st.file_uploader("Upload customer audio (WAV/MP4)", type=["wav", "mp4"])

if st.button("Process Audio") and uploaded_file:
    with st.spinner("Processing..."):
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # Process using the stateful agent
            result = process_audio_file(
                audio_path=tmp_path,
                output_dir="sample_outputs/",
                agent=st.session_state.agent,
                retriever=st.session_state.retriever
            )

            st.write("**Customer Transcript:**", result.get("transcript"))
            st.write("**Agent Response:**", result.get("response"))
            
            if result.get("reply_audio"):
                st.audio(result.get("reply_audio"))
            
            with st.expander("View Evidence & Data"):
                st.json(result.get("retrieved_evidence", []))
                st.json(result.get("tool_calls", []))

        except Exception as e:
            st.error(f"Error: {type(e).__name__}: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)