# ZentroMart VoiceOps Assistant

**GitHub Repository:** [https://github.com/Pjdey28/VoiceOps-Zentromart](https://github.com/Pjdey28/VoiceOps-Zentromart)

This project implements the four capstone stages:

1. Prompt-only LLM answers.
2. Authoritative knowledge-base retrieval augmented generation (RAG).
3. Multi-turn conversation handling with verified order lookup and support-ticket tools.
4. Gnani.ai-only voice processing with speech-to-text, grounded replies, and text-to-speech.

## Setup

1. Extract the supplied `Capstone_Project_Dataset_FTTT_2026` directory next to this
   project, or set `DATASET_DIR` to its absolute path.
2. Create a virtual environment and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set the required credentials:

   ```env
   GROQ_API_KEY=primary_groq_key
   GROQ_API_KEY_2=optional_fallback_groq_key
   GROQ_MODEL=llama-3.3-70b-versatile
   GNANI_API_KEY=gnani_key
   DATASET_DIR=../Capstone_Project_Dataset_FTTT_2026
   ```

   Never commit `.env`, API keys, the dataset, or `.venv`.

## Exact Run Command (For Evaluators)

The required batch command takes a folder of customer audio files, processes every file with Gnani STT and Gnani TTS, and saves its replies (as text and audio) with the document chunks it used:

```powershell
python run.py voice-batch --input .\audio --output .\outputs\voice
```

The voice pipeline accepts WAV and MP4 files up to 60 seconds. MP4 audio is decoded with PyAV and converted to a temporary mono 16 kHz WAV before being sent to Gnani STT. FFmpeg is an optional fallback when PyAV is unavailable. It uses `en-IN` (with seamless `hi-en` Hinglish fallback), the `Kaveri` voice, and the explicitly named `timbre-v2.5` TTS model by default.

## Evaluation Commands

Run these from the project root to evaluate the pipeline stages:

```powershell
python run.py check-data
python run.py inspect
python run.py baseline --limit 5
python run.py rag-evaluate
python run.py conversation-evaluate-v2
python -m src.test_conversation
```

Evaluation outputs are written under `outputs/`.

The conversion-ready submission report is [SUBMISSION_REPORT.md](SUBMISSION_REPORT.md). Convert it to the required `report.pdf` after the final evaluation run and keep it within four pages.

## Interactive Streamlit conversation

Launch the conversation UI with:

```powershell
streamlit run app.py
```

The interface accepts either typed messages or WAV/MP4 audio turns and keeps
follow-up messages in the same conversation. Use **Reset conversation** to
clear the assistant state. Each turn displays the customer transcript, the
assistant response, expandable document evidence, verified support actions,
and a playable WAV reply when TTS is available. The UI renders these results
as readable conversation content rather than JSON; JSON files remain available
only for the batch evaluation artifact workflow.

## Voice batch

The required batch command accepts a folder of WAV or MP4 files, processes every file
with Gnani STT and Gnani TTS, and writes a manifest plus one set of transcript,
response, evidence, reply-audio, and result files per input:

```powershell
python run.py voice-batch --input .\audio --output .\outputs\voice
```

## Data and safety behavior

- Current authoritative policy documents are preferred over older guides, FAQs,
  and marketing content.
- Retrieved evidence is recorded with every RAG and voice result.
- The assistant does not disclose order details until the supplied lookup tool
  verifies the caller with a phone suffix or delivery pincode.
- Failed ticket creation is reported as unconfirmed; the assistant does not
  claim a handover succeeded without a positive tool result.
- Unsupported policy questions receive an uncertainty response rather than an
  invented policy.

Run the supplied dataset validator from the extracted dataset directory:

```powershell
python validate_dataset.py
```
