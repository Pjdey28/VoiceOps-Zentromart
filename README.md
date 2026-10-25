# ZentroMart VoiceOps Assistant

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

## Commands

Run these from the project root:

```powershell
python run.py check-data
python run.py inspect
python run.py baseline --limit 5
python run.py rag-evaluate
python run.py conversation-evaluate-v2
python -m src.test_conversation
```

Evaluation outputs are written under `outputs/`.

## Voice batch

The required batch command accepts a folder of WAV or MP4 files, processes every file
with Gnani STT and Gnani TTS, and writes a manifest plus one set of transcript,
response, evidence, reply-audio, and result files per input:

```powershell
python run.py voice-batch --input .\audio --output .\outputs\voice
```

The voice pipeline accepts WAV and MP4 files up to 60 seconds. MP4 input is
converted to a temporary mono WAV using FFmpeg before being sent to Gnani STT.
Install FFmpeg separately and make sure `ffmpeg` is available on `PATH`.
It uses `en-IN`, the
`Kaveri` voice, and the explicitly named `timbre-v2.5` TTS model by default.

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
