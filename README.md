# ZentroMart VoiceOps Assistant
## Project links
- **Repository:** https://github.com/Pjdey28/VoiceOps-Zentromart
- **Dataset:** `https://drive.google.com/drive/folders/1D_Oj5YdpC_pXLB6dSciCBqwLnWnouMc4?usp=sharing`
- **Deployed application:** `DEPLOYED_LINK_TO_BE_ADDED`
- **Submission report:** [report.pdf](report.pdf)
- **Demo video:** [demo.mp4](demo.mp4)
## Overview
ZentroMart VoiceOps is a customer-support assistant for an e-commerce
operation.
It answers policy questions.
It retrieves evidence from company documents.
It uses product-catalog facts for product-specific questions.
It manages multi-turn conversations.
It verifies order information before disclosing private details.
It creates support tickets only when escalation is appropriate.
It accepts customer voice messages.
It supports WAV and MP4 input.
It transcribes audio with Gnani.ai speech-to-text.
It generates a grounded answer.
It synthesizes a reply with Gnani.ai text-to-speech.
It exposes both a command-line batch workflow and a Streamlit conversation UI.
The project is designed to make evidence, uncertainty, tool actions, and
limitations visible during normal use and evaluation.
## Product goals
The assistant has five primary goals.
### Goal 1: Answer accurately
Policy answers should be based on current authoritative documents.
Product-specific answers should combine catalog facts with applicable policy.
The assistant should acknowledge uncertainty when evidence is insufficient.
### Goal 2: Preserve conversation context
Customers often provide information across several turns.
For example, one turn may contain an order ID.
The next turn may contain the delivery pincode.
The following turn may ask for the order status.
The controller retains the relevant state across those turns.
### Goal 3: Protect customer information
An order ID alone does not authorize disclosure of private order details.
The controller uses the official lookup tool and verification value.
The assistant does not claim verification without a positive tool result.
### Goal 4: Support voice interaction
Customers can provide an audio file instead of typing.
The system transcribes the audio.
The resulting transcript follows the same conversation path as typed input.
The response can be synthesized back to audio.
### Goal 5: Make results traceable
Every important answer should be explainable through retrieved evidence.
Batch mode saves structured artifacts.
Interactive mode presents the same information as readable conversation content.
## Capstone coverage
The project implements four stages.
### Stage 1: Prompt-only evaluation
The baseline sends practice questions to the language model without retrieval.
This establishes a comparison point.
The baseline is useful because retrieval quality should be measured against a
known non-retrieval approach.
### Stage 2: Retrieval-augmented generation
The RAG system searches authoritative knowledge chunks.
The system includes policy documents, support material, and product records.
Retrieved context is inserted into a grounded answer prompt.
### Stage 3: Multi-turn support conversations
The conversation controller chooses between answering, looking up an order, and
creating a support ticket.
The controller also applies deterministic safety rules around order IDs and
verification.
### Stage 4: Voice processing
The voice pipeline accepts audio files.
It calls Gnani STT for transcription.
It sends the transcript through the assistant.
It calls Gnani TTS for the response.
It saves transcript, response, evidence, and reply audio artifacts.
## Repository map
The root runner is [run.py](run.py).
The Streamlit interface is [app.py](app.py).
Configuration is in [src/config.py](src/config.py).
Dataset loading is in [src/dataset.py](src/dataset.py).
Knowledge-base parsing is in [src/knowledge_base.py](src/knowledge_base.py).
Retrieval and ranking are in [src/retrieval.py](src/retrieval.py).
Grounded RAG generation is in [src/rag.py](src/rag.py).
Conversation state and tool policy are in [src/conversation.py](src/conversation.py).
Language-model access is in [src/llm.py](src/llm.py).
Order and ticket tools are in [src/tools.py](src/tools.py).
Voice processing is in [src/voice.py](src/voice.py).
Evaluation runners are in [src/evaluate.py](src/evaluate.py).
Conversation result inspection is in
[src/inspect_conversation_results.py](src/inspect_conversation_results.py).
The sample conversation smoke test is in
[src/test_conversation.py](src/test_conversation.py).
The Streamlit watcher configuration is in
[.streamlit/config.toml](.streamlit/config.toml).
The dependency list is in [requirements.txt](requirements.txt).
The safe configuration template is in [.env.example](.env.example).
## Architecture
The system has a layered architecture.
### Input layer
Input can be typed text.
Input can be WAV audio.
Input can be MP4 audio.
The CLI accepts a directory for batch processing.
The Streamlit app accepts one turn at a time.
### Normalization layer
Audio is normalized before STT.
MP4 audio is decoded with PyAV when available.
The normalized format is mono, 16 kHz, signed 16-bit WAV.
WAV files are validated for readability and duration.
The maximum clip duration is 60 seconds.
### Language layer
Gnani STT produces the customer transcript.
The transcript is passed into the conversation controller.
The controller may call retrieval and support tools.
Gnani TTS produces the spoken assistant response.
### Evidence layer
The retriever returns knowledge chunks with similarity scores.
The answer prompt includes source identifiers and chunk text.
The UI shows the evidence in expandable sections.
The batch workflow saves evidence in a JSON artifact.
### State layer
The conversation controller stores messages.
It stores a pending order ID when appropriate.
It stores verified order data after successful verification.
It stores the last tool result and per-turn telemetry.
The Streamlit session stores the rendered conversation turns.
## Local setup
### Prerequisites
Use Python 3.10 or newer.
Python 3.14 has been used during local development.
Use a terminal opened at the repository root.
The supplied practice dataset must be available.
Internet access is needed for hosted model and Gnani calls.
### Create the virtual environment
On Windows PowerShell:
```powershell
py -m venv .venv
```
Activate it:
```powershell
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, use the Python executable directly.
### Install dependencies
```powershell
pip install -r requirements.txt
```
The dependency list includes:
- `groq` for language-model calls.
- `python-dotenv` for local configuration.
- `pandas` for CSV loading.
- `numpy` and `scikit-learn` for retrieval support.
- `sentence-transformers` and `torch` for embedding-based retrieval support.
- `gnani-vachana` for Gnani STT and TTS.
- `av` for local MP4 decoding.
- `streamlit` for the interactive UI.
### Configure the environment
Copy the template:
```powershell
Copy-Item .env.example .env
```
Set values in `.env`:
```env
GROQ_API_KEY=primary_groq_key
GROQ_API_KEY_2=optional_fallback_groq_key
GROQ_MODEL=openai/gpt-oss-120b
GNANI_API_KEY=gnani_key
DATASET_DIR=../Capstone_Project_Dataset_FTTT_2026
```
Never commit `.env`.
Never paste API keys into the README.
Never include API keys in screenshots.
Never include the private dataset in the repository unless its license permits
redistribution.
## Dataset configuration
The application reads `DATASET_DIR` from the environment.
Relative dataset paths are resolved relative to the project root.
An absolute path can be used when the dataset is stored elsewhere.
Example:
```env
DATASET_DIR=C:\Users\name\Downloads\Capstone_Project_Dataset_FTTT_2026
```
CSV identifiers are loaded as strings.
This preserves leading zeroes in order IDs.
It also preserves SKU values and pincode values.
Run the dataset check:
```powershell
python run.py check-data
```
Inspect the practice schemas:
```powershell
python run.py inspect
```
Run the supplied validator from the dataset directory:
```powershell
python validate_dataset.py
```
If the dataset link is public, add it to the project links section above.
If access requires authentication, document the access instructions without
publishing credentials.
## Command-line workflows
### Show available commands
```powershell
python run.py --help
```
### Validate data
```powershell
python run.py check-data
```
### Inspect practice data
```powershell
python run.py inspect
```
### Run a small baseline
```powershell
python run.py baseline --limit 5
```
### Run the full baseline
```powershell
python run.py baseline
```
### Run RAG evaluation
```powershell
python run.py rag-evaluate
```
### Run conversation evaluation
```powershell
python run.py conversation-evaluate-v2
```
### Run the conversation smoke test
```powershell
python -m src.test_conversation
```
### Inspect conversation mismatches
```powershell
python -m src.inspect_conversation_results
```
## Interactive Streamlit application
Launch the UI:
```powershell
streamlit run app.py
```
The app opens a conversation-style support window.
Type a customer message in the chat input.
The assistant response appears as a new chat turn.
The controller remains in session state.
Send a follow-up without repeating the earlier information.
Use **Reset conversation** to clear the controller state.
### Text turn behavior
The typed message is sent directly to the controller.
The application retrieves evidence for the message.
The controller generates the response.
The application attempts to synthesize the response.
The response appears as readable text.
The evidence appears in an expandable section.
The reply audio appears in an audio player when TTS succeeds.
### Audio turn behavior
Upload a WAV file or MP4 file.
Click **Send audio message**.
The app decodes MP4 audio if necessary.
Gnani STT produces a transcript.
The transcript is displayed as the customer message.
The same transcript is processed by the conversation controller.
The assistant response is displayed below it.
The uploaded input audio can be played back.
The synthesized assistant reply can also be played.
### Human-readable output
The interactive app does not show raw JSON to the customer.
Evidence is rendered as readable document sections.
Tool actions are rendered as readable support actions.
JSON artifacts remain available for machine evaluation and reproducibility.
## Voice batch workflow
The evaluator-facing command is:
```powershell
python run.py voice-batch --input .\audio --output .\outputs\voice
```
The input folder may contain WAV files.
The input folder may contain MP4 files.
Unsupported extensions are ignored.
Every supported file is processed independently.
A fresh conversation controller is used for each batch file.
The batch creates `batch_manifest.json`.
Each successful file creates a transcript text file.
Each successful file creates a response text file.
Each successful file creates an evidence JSON file.
Each successful file creates a reply WAV file.
Each file also receives a result JSON artifact.
The manifest records successful and failed counts.
The batch continues processing after an individual file failure.
This is useful for organizer-side testing because one bad input does not hide
the results for all other inputs.
## MP4 processing
MP4 is a container and may contain video as well as audio.
The application ignores the video stream.
PyAV extracts the first available audio stream.
The audio is resampled to mono 16 kHz signed 16-bit PCM.
Gnani STT receives the temporary WAV path.
Temporary conversion files are removed after transcription.
The project does not require a separate FFmpeg installation when PyAV is
installed successfully.
FFmpeg is an optional fallback.
If an MP4 contains no audio stream, processing fails with a clear error.
If the audio cannot be decoded, processing fails with a clear error.
The 60-second clip limit applies after conversion.
## Retrieval design
The knowledge base is loaded into structured `KnowledgeChunk` records.
Markdown documents are split into meaningful sections.
Product rows are added as structured catalog chunks.
Product chunks use identifiers such as `products.csv:<SKU>`.
The retriever returns a `RetrievedChunk` wrapper.
The wrapper contains the knowledge chunk and a similarity score.
The title, section ID, source file, category, status, and text remain available.
### Product-aware retrieval
A product question may require two types of evidence.
The first type is a catalog fact.
Examples include SKU, category, seller, brand, or assured status.
The second type is a policy rule.
Examples include return eligibility, warranty conditions, or cancellation rules.
The retriever boosts exact SKU matches.
It also considers exact product-title matches.
The RAG flow preserves policy and catalog evidence diversity.
This prevents a row from `products.csv` from crowding out the policy section
needed to interpret that product.
### Authority ordering
Current authoritative policy documents outrank older guides.
Historical support logs are not treated as current policy.
Marketing content is not treated as an authoritative rule.
The prompt explicitly instructs the model to prefer current evidence.
## Grounded answer behavior
RAG answers receive retrieved context with source identifiers.
The system prompt prohibits invented policy details.
The system prompt prohibits unsupported timelines and amounts.
The system prompt requires an uncertainty response when evidence is insufficient.
The prompt asks for a final Sources line.
The answer is not considered reliable merely because it sounds confident.
Reviewers should inspect both the answer and its evidence.
An answer that lacks evidence should be treated as a quality defect.
An answer that contradicts current policy should be treated as a quality defect.
## Conversation controller
The controller is implemented by `ConversationController`.
It stores customer and assistant messages.
It resets per-turn telemetry before handling a new message.
It tracks the last action.
It tracks the normalized decision.
It tracks the last tool calls.
It tracks verification failures.
It tracks a pending order ID.
It tracks the verified order after successful verification.
### Deterministic order extraction
The controller recognizes eight-digit order IDs.
It supports spaces and hyphens between digits.
It supports explicitly labelled order IDs.
Identifiers remain strings.
This avoids losing leading zeroes.
The deterministic extractor prevents the model from accidentally choosing an
order number from unrelated text.
### Available actions
The language-model decision schema has three primary actions.
`answer` handles general questions.
`lookup_order` uses the official order lookup tool.
`create_ticket` escalates to human support.
The controller normalizes action labels for evaluation.
### Clarification behavior
Greetings and vague requests receive a concise clarification.
The assistant does not immediately search for an arbitrary policy answer.
The assistant does not create a ticket for a greeting.
The assistant asks for enough information to proceed.
## Order privacy and verification
Order details are private information.
An order ID alone is not sufficient verification.
The user may provide the registered phone suffix.
The user may provide the delivery pincode.
The lookup tool decides whether verification succeeded.
The controller only discloses details after a positive result.
A failed verification is not represented as a successful lookup.
Repeated failures trigger a refusal path.
The assistant directs the customer to official support after repeated failures.
The model is not allowed to override this controller policy.
## Ticket creation
A ticket may be appropriate when the customer explicitly requests a human.
The controller calls the official ticket tool.
The assistant does not claim a ticket was created before confirmation.
If the tool fails, the response says that creation could not be confirmed.
This avoids a common support-agent failure mode: claiming a handover that never
occurred.
## LLM configuration and rate limits
The language-model client reads `GROQ_API_KEY`.
An optional `GROQ_API_KEY_2` can act as a fallback.
The primary key is attempted first.
Rate-limit errors are retried with bounded backoff.
Provider retry hints are respected when available.
The fallback key is used after the primary key exhausts its retry path.
The code does not silently convert arbitrary provider errors into successful
answers.
Only the key names are documented.
Actual secrets must remain in `.env`.
## Evaluation methodology
The baseline evaluates prompt-only behavior.
The RAG evaluation measures evidence-grounded behavior.
The conversation evaluation compares expected and actual tool actions.
The voice workflow is validated separately because live Gnani credentials and
audio access may not be available in every environment.
### RAG metrics
The evaluator records must-include behavior.
It records must-not-include behavior.
It records gold-evidence recall.
It records uncertainty correctness.
It records grounded-answer rates.
The output is written under `outputs/rag/`.
### Conversation metrics
The evaluator records runtime errors.
It records tool-call agreement.
It records comparable decision agreement.
It stores mismatch details for inspection.
The output is written under `outputs/conversations/`.
### Honest validation
Live voice success should not be claimed without a real Gnani run.
MP4 decoding can be tested locally without sending audio to Gnani.
TTS output requires valid Gnani credentials.
The README deliberately distinguishes local conversion validation from
end-to-end voice validation.
## Output artifacts
The `outputs/` directory contains evaluation results.
The baseline summary is stored under `outputs/baseline/`.
The RAG summary is stored under `outputs/rag/`.
Conversation summaries are stored under `outputs/conversations/`.
Voice batch outputs can be stored under `outputs/voice/`.
The Streamlit demo stores conversation media under
`sample_outputs/conversation/`.
Generated artifacts should be reviewed before packaging.
Do not package credentials.
Do not package the virtual environment.
Do not package private datasets unless explicitly permitted.
## Error handling
Missing dataset directories produce an explicit validation error.
Missing model credentials produce an explicit configuration error.
Missing Gnani credentials produce an explicit voice error.
Unsupported audio extensions produce an explicit format error.
Unreadable WAV files produce an explicit validation error.
MP4 files without audio streams produce an explicit decoding error.
Empty STT transcripts produce an explicit transcription error.
Empty TTS responses produce an explicit synthesis error.
Batch failures are recorded per file.
The batch manifest reports the final success and failure counts.
The interactive UI shows the exception type and message.
## Security considerations
Secrets are loaded from environment variables.
The `.env` file is excluded from version control.
The assistant does not reveal hidden prompts.
The assistant does not reveal unverified order details.
The assistant does not invent successful tool results.
Uploaded audio files are written to controlled output paths.
Temporary conversion files are removed after use.
Output filenames are sanitized.
The application should still be deployed behind HTTPS.
Production deployment should use a secrets manager.
Production logs should avoid storing unnecessary personal data.
Production retention rules should be defined before storing recordings.
## Usage walkthrough
Open the deployed link.
Send: `What is the return policy?`
Read the response.
Expand the evidence.
Check whether the answer cites a current policy section.
Send a follow-up about a particular product.
Check whether catalog and policy evidence are both relevant.
Reset the conversation.
Upload `audio/policy.mp4`.
Check the transcript.
Play the customer input audio.
Read the assistant response.
Play the assistant reply audio if credentials are configured.
Try an order ID without verification.
Confirm that private details are not disclosed.
Provide a valid verification value from the test data.
Confirm that the tool result controls the disclosure.
Ask for a human agent.
Confirm that ticket creation is not claimed without positive confirmation.
## Design decisions
### Why is retrieval needed?
A language model can produce fluent but unsupported policy details.
Retrieval supplies the company-specific evidence required for grounded answers.
### Why include product rows?
Policy alone may not identify whether a product is eligible.
The product catalog supplies SKU, category, seller, and warranty facts.
### Why use deterministic order extraction?
Order identifiers have a strict format.
A deterministic extractor is more reliable than asking a model to infer the ID.
### Why keep tool calls in the controller?
Privacy and authorization rules should not depend only on generated text.
The controller can enforce the verification boundary before formatting a reply.
### Why keep JSON artifacts?
Machine-readable artifacts support evaluation and debugging.
The UI presents the same information in a human-readable form.
### What would be improved for production?
Use a persistent conversation store.
Use authenticated users.
Use a production ticketing integration.
Add observability and tracing.
Add latency and cost metrics.
Add stronger audio abuse controls.
Add automated regression tests for policy changes.
## Known limitations
The current tools are project support tools rather than a live commerce backend.
Voice success depends on Gnani credentials and service availability.
TTS output cannot be verified without a live Gnani call.
MP4 decoding supports audio streams but does not transcribe video content.
The Streamlit session state is not a durable production conversation database.
The local output directory may contain customer-provided audio.
Evaluation agreement is not the same as real-world customer satisfaction.
Model provider rate limits can affect full evaluation runs.
The deployment link must be tested separately from local development.
The dataset link and deployed link are intentionally placeholders until the
submission owner supplies the final URLs.
## Design principles
This project prioritizes controlled behavior over unsupported confidence.
The assistant is expected to say that it cannot confirm something when the
documents do not support it.
The assistant is expected to ask for verification before disclosing an order.
The assistant is expected to expose evidence for policy answers.
The assistant is expected to report failed tools honestly.
The assistant is expected to preserve context across follow-up turns.
The command-line workflow is designed for repeatable organizer evaluation.
The Streamlit workflow is designed for interactive product demonstration.
The report provides deeper implementation detail than this README.
The README is intentionally structured as a review guide.
The three external links at the top should be completed before submission.
