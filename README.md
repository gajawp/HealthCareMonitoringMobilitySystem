<div align="center">

<strong>HealthBridge AI</strong>

<strong>An Inclusive and Secure Conversational Assistant for Remote Patient Mobility Monitoring</strong>

CS 5100 — Foundations of Artificial Intelligence
Northeastern University | 2026

</div>

HealthBridge AI is a retrieval-augmented conversational prototype that helps patients, caregivers, and clinicians understand longitudinal mobility-monitoring records through natural-language questions.

The system combines deterministic authentication, patient-level authorization, structured data retrieval, identifier protection, multilingual large-language-model (LLM) generation, optional voice interaction, and safety validation. Patient-specific measurements are retrieved from stored records rather than generated from the model's memory.

Important: HealthBridge AI is an academic prototype. It explains recorded mobility information and approved exercise guidance; it does not diagnose conditions, prescribe treatment, replace clinical judgment, or claim production HIPAA compliance.

<strong>Table of Contents</strong>

Key Features

System Architecture

How a Request Is Processed

Authorization Model

Technology Stack

Project Structure

Data

Exact Steps to Run the Project

Using the Application

Testing and Evaluation

Results

Security and Privacy Design

Limitations

Future Improvements

Team Contributions

Repository Practices

Acknowledgments

<strong>Key Features</strong>

Natural-language access to recorded mobility-session data

Exact retrieval of repetitions, laterality, duration, movement-quality statistics, and flagged events

Support for latest-session, previous-session, date-specific, summary, and paraphrased questions

Role-based access for patients, caregivers, and clinicians

Patient-level assignment checks before data retrieval

Retrieval-augmented responses grounded in structured records

Separate retrieval of allow-listed exercise guidance

Patient-identifier tokenization before model-facing context is built

Multilingual questions and responses

Optional speech-to-text and text-to-speech interaction

Resizable Streamlit chat interface with synchronized text and audio

Deterministic responses for authorization failures, missing data, and protected requests

Audit-friendly modular architecture

Automated factual, multilingual, authorization, and latency evaluation

<strong>System Architecture</strong>

flowchart TD
    UI["Streamlit UI<br/>Text or voice"] --> AUTH["Authentication and<br/>patient-scope authorization"]
    AUTH -->|Denied| DENY["Deterministic denial"]
    AUTH -->|Allowed| INTENT["Intent and temporal<br/>query resolution"]
    INTENT --> MR["Structured mobility<br/>retrieval"]
    INTENT --> KR["Approved knowledge<br/>retrieval"]
    MR --> CTX["Minimal de-identified<br/>context builder"]
    KR --> CTX
    CTX --> LLM["Grounded multilingual<br/>LLM generation"]
    LLM --> SAFE["Privacy and safety<br/>validation"]
    SAFE --> OUT["Text response and<br/>optional speech"]

The architecture intentionally separates deterministic responsibilities from generative behavior:

Application code decides whether access is permitted.

Structured retrieval selects patient-specific facts.

The context builder minimizes and de-identifies evidence.

The LLM explains authorized evidence in the selected language.

Guardrails validate the response before it reaches the user.

<strong>How a Request Is Processed</strong>

The interface receives the question, selected language, authenticated user, role, and selected patient.

The session is validated.

The authorization service verifies that the user may access the selected patient.

The intent module identifies the request type and resolves expressions such as latest, previous, or a specific date.

The mobility retriever loads exact values from structured records.

When appropriate, the knowledge retriever loads only approved exercise information.

Direct identifiers are replaced with internal tokens, and a minimal evidence context is constructed.

The LLM converts the evidence into a concise response in the selected language.

Privacy and safety checks validate the response.

The interface returns text and, when enabled, synthesized speech.

Authorization failures, unavailable records, and protected requests use deterministic response templates instead of relying on the LLM.

<strong>Authorization Model</strong>

Role

Permitted scope

Typical capabilities

Patient

Own record only

Ask about personal mobility sessions and summaries

Caregiver

Explicitly assigned patients

View assigned patient records and approved guidance

Clinician

Patients within configured scope

Review authorized patient records and approved guidance

Role permission alone is not sufficient. Caregiver and clinician requests must also pass the patient-assignment check. Authorization occurs before retrieval, preventing unauthorized patient data from entering model-facing context.

<strong>Technology Stack</strong>

Component

Technology

Purpose

User interface

Python, Streamlit

Dashboard integration, chat widget, language selection, and voice controls

Application logic

Modular Python services

Intent detection, orchestration, retrieval, context construction, and guardrails

Generative AI

Configured LLM API

Grounded explanation and multilingual response generation

Authentication

bcrypt and session services

Password hashing and authenticated sessions

Authorization

Role-based and patient-scope checks

Record-level access enforcement

Data

Structured local JSON/session records

Mobility measurements and demo assignments

Voice

Speech recognition and synthesis services

Optional spoken input and output

Evaluation

Python test runner and CSV suite

Component tests, factual accuracy, authorization accuracy, and latency

<strong>Project Structure</strong>

The main application modules are organized by responsibility:

HealthCareMonitoringMobilitySystem/
├── main.py                    # Streamlit application entry point
├── chatbot_ui.py              # Chat panel, multilingual UI, voice, and resizing
├── orchestrator.py            # Coordinates the complete request workflow
├── intent.py                  # Intent classification and query interpretation
├── mobility_retriever.py      # Date-specific and aggregate mobility retrieval
├── context_builder.py         # Minimal de-identified prompt context
├── llm_service.py             # LLM request and grounded response generation
├── guardrails.py              # Privacy, safety, and unsupported-request checks
├── login_security.py          # Login and credential-security helpers
├── service.py                 # Authentication, authorization, and audit services
├── session.py                 # Session-state management
├── models.py                  # Application data models
├── demo_users.py              # Synthetic users and role assignments
├── config.py                  # Application configuration
├── requirements.txt           # Python dependencies
├── tests/                     # Component and integration tests
└── data/                      # Structured mobility and approved knowledge data

<strong>Data</strong>

The prototype uses local structured mobility-session records produced by the monitoring dashboard and synthetic user-to-patient assignments for evaluation. No public clinical dataset or pretrained patient model is required.

A session record can include:

Session identifier and timestamp

Total repetitions

Left- and right-side repetition counts

Frame count

Flagged movement events

Duration statistics

Movement-quality or jerk statistics

Preprocessing validates required fields, parses timestamps, normalizes numeric values, orders sessions chronologically, computes or loads summaries, and maps patient identifiers to internal tokens.

All bundled demonstration data should be synthetic or de-identified. Do not commit protected health information (PHI) or real patient credentials.

<strong>Exact Steps to Run the Project</strong>

<strong>Prerequisites</strong>

Python 3.10 or later

pip

OpenAI API key

Git

A microphone and audio output device for optional voice features

<strong>Step 1 — Clone the repository</strong>

git clone https://github.com/gajawp/HealthCareMonitoringMobilitySystem.git

<strong>Step 2 — Open the project directory</strong>

cd HealthCareMonitoringMobilitySystem

<strong>Step 3 — Create a virtual environment</strong>

<strong>macOS or Linux</strong>

python3 -m venv venv

<strong>Windows PowerShell</strong>

python -m venv venv

<strong>Step 4 — Activate the virtual environment</strong>

<strong>macOS or Linux</strong>

source venv/bin/activate

<strong>Windows PowerShell</strong>

.\venv\Scripts\Activate.ps1

After activation, the terminal prompt should begin with (venv).

<strong>Step 5 — Install all required dependencies</strong>

python3 -m pip install --upgrade pip
pip install -r requirements.txt

On Windows, use python instead of python3 if necessary.

<strong>Step 6 — Create the environment file</strong>

Create a file named .env in the project root—the same directory that contains main.py and requirements.txt.

Add the following line:

OPENAI_API_KEY=your_actual_openai_api_key

The project should now contain:

HealthCareMonitoringMobilitySystem/
├── .env
├── main.py
├── requirements.txt
└── ...

Security warning: Never commit the .env file, API keys, passwords, tokens, or patient information to GitHub.

<strong>Step 7 — Start the Streamlit application</strong>

streamlit run main.py

If the streamlit command is not recognized, run:

python3 -m streamlit run main.py

On Windows:

python -m streamlit run main.py

<strong>Step 8 — Open the application</strong>

Streamlit should open the application automatically. If it does not, open this address in a browser:

http://localhost:8501

<strong>Step 9 — Sign in and use the assistant</strong>

Sign in using one of the synthetic accounts configured in demo_users.py.

Select an authorized patient if the signed-in role supports patient selection.

Select a language.

Open the assistant panel.

Type a mobility question or select the microphone button.

Review the grounded response or enable audio playback.

<strong>Step 10 — Stop the application</strong>

Return to the terminal and press:

Ctrl + C

<strong>Quick Start — macOS or Linux</strong>

git clone https://github.com/gajawp/HealthCareMonitoringMobilitySystem.git
cd HealthCareMonitoringMobilitySystem
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py

Before the final command, create .env in the project root and add OPENAI_API_KEY=your_actual_openai_api_key.

<strong>Quick Start — Windows PowerShell</strong>

git clone https://github.com/gajawp/HealthCareMonitoringMobilitySystem.git
cd HealthCareMonitoringMobilitySystem
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py

Before the final command, create .env in the project root and add OPENAI_API_KEY=your_actual_openai_api_key.

<strong>Common Setup Problems</strong>

Problem

Solution

python3: command not found

Use python instead of python3.

streamlit: command not found

Activate venv, reinstall requirements.txt, or use python -m streamlit run main.py.

ModuleNotFoundError

Confirm the virtual environment is active and rerun pip install -r requirements.txt.

OpenAI authentication error

Confirm .env is in the project root and OPENAI_API_KEY contains a valid key without quotes.

Microphone is unavailable

Allow microphone permission in the browser and confirm an input device is connected.

Port 8501 is already in use

Run streamlit run main.py --server.port 8502.

<strong>Using the Application</strong>

Open the Streamlit URL in a browser.

Sign in with a configured synthetic demo account.

Select an authorized patient when the role permits patient selection.

Choose the chat language.

Enter a question or select the microphone icon for voice input.

Enable spoken output if desired.

Review the grounded answer and any no-data, safety, or authorization message.

Example questions:

How many repetitions were completed in the latest session?
How many were on the left and right sides?
Show the session from March 14, 2026.
Compare the latest session with the previous session.
Were any movements flagged?
Summarize the patient's recent mobility progress.
Is there data for January 20, 2026?
What approved exercise information is available?

The same types of questions can be asked in the languages supported by the configured interface and model.

<strong>Testing and Evaluation</strong>

<strong>Component Tests</strong>

From the project root, run:

pytest -v

The component tests cover areas such as:

Mobility retrieval

Latest, previous, and date-specific temporal behavior

Patient-scope authorization

Identifier protection

No-data fallbacks

Privacy-sensitive and unsupported requests

<strong>Automated CSV Evaluation</strong>

Run the repository's evaluation script against the included test-case CSV. For example:

python evaluate.py

If the evaluation script in your branch has a different filename, run that script from the project root. The evaluation CSV records the prompt, expected source values, category, language, authorization expectation, response, correctness, and end-to-end latency.

The metrics are calculated as:

Factual accuracy = correct factual cases / total factual cases × 100
Authorization accuracy = correct allow-or-deny decisions / authorization cases × 100
Mean latency = sum of end-to-end response times / total test cases

<strong>Results</strong>

<strong>Overall Automated Evaluation</strong>

Measure

Result

Total test cases

100

Execution errors

0

Factual accuracy

87/90 (96.67%)

Authorization accuracy

10/10 (100.00%)

Mean end-to-end latency

4.033 seconds

Non-English language accuracy

100% in each of five evaluated language groups

<strong>Factual Accuracy by Category</strong>

Category

Correct / Total

Accuracy

Date-specific

14/15

93.33%

Direct factual

20/20

100.00%

Multilingual

30/30

100.00%

Paraphrase

13/15

86.67%

Summary

10/10

100.00%

The three factual errors were concentrated in date interpretation and paraphrase handling rather than direct structured-data lookup.

<strong>Component and Security Validation</strong>

Test area

Observed result

Mobility retrieval

Exact values returned for latest, date-specific, and summary queries

Authorization

Nine component scenarios and ten end-to-end cases passed

Identifier protection

Direct patient identifiers excluded from model-facing prompts

Temporal handling

Specific dates and relative references resolved to stored sessions

No-data behavior

Stable deterministic response returned when records were unavailable

Privacy-sensitive requests

Protected or out-of-scope requests were blocked

<strong>Formative User Study</strong>

The prototype was evaluated by six participants:

Voice interaction achieved a construct median of 5.0.

Response clarity, accessibility, and language experience each achieved a median of 4.0.

Response speed achieved a mean and median of 4.5.

Error and no-data messaging achieved a mean and median of 4.5.

Microphone discoverability scored lower at a mean and median of 3.5, identifying a clear interface improvement.

Overall helpfulness ratings ranged from 4 to 5.

These results demonstrate technical and usability feasibility; they do not establish clinical effectiveness.

<strong>Security and Privacy Design</strong>

HealthBridge AI uses defense in depth:

Passwords are hashed with bcrypt.

Authentication is required before protected workflows.

Role and patient-assignment checks run before retrieval.

Access decisions are made by application code, not the LLM.

Retrieval is limited to structured records and approved knowledge sources.

Direct patient identifiers are replaced with internal tokens.

Model context is minimized to the facts needed for the request.

Privacy and safety filters validate generated responses.

Unauthorized, missing-data, and protected states use deterministic templates.

Sensitive actions can be recorded through audit logging.

For a production deployment, additional work would be required, including formal threat modeling, encrypted storage and transport, secrets management, penetration testing, monitoring, retention policies, clinical governance, and legal/compliance review.

<strong>Limitations</strong>

The formative user study included only six participants.

Evaluation used a local prototype dataset and synthetic role assignments.

External clinical datasets and real deployment conditions were not evaluated.

Speech recognition and synthesis were not benchmarked using word-error rate or standardized audio tests.

Prompt-injection testing primarily covered direct attacks; paraphrased, multilingual, indirect, and multi-turn attacks require a larger adversarial set.

Grounding was evaluated through agreement with source values rather than full manual annotation of every response claim.

The system is an informational prototype and not a medical device.

<strong>Future Improvements</strong>

Add grammar-based temporal parsing and a larger paraphrase corpus.

Display evidence citations that link responses to session and metric sources.

Expand multilingual clinical-safety and accessibility evaluation.

Conduct a larger study with patients, caregivers, and clinicians.

Benchmark speech recognition and synthesis formally.

Add streaming responses, caching, and asynchronous speech synthesis.

Expand red-team testing for prompt injection and sensitive-data disclosure.

Add production-grade key management, encryption, monitoring, and compliance review.

<strong>Team Contributions</strong>

Contributor

Primary contributions

Preethi Gajawada

System integration; secure RAG architecture; intent, retrieval, privacy, and guardrail workflow; multilingual and voice interface; automated evaluation; analysis; documentation

Madhu Babu Cherukuri

Dashboard and mobility-data integration; implementation support; test-case review; interface validation; results review; presentation and documentation support

Dr. Sarita Singh

Faculty guidance, project review, academic feedback, and evaluation direction

<strong>Repository Practices</strong>

Before pushing changes, confirm that the repository does not contain secrets or sensitive data:

.env
.env.*
!.env.example
venv/
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.DS_Store

Recommended checks:

git status
git diff --cached

Do not push real patient records, API credentials, raw authentication logs, or exported audio containing sensitive information.

<strong>Acknowledgments</strong>

This project was developed for CS 5100 – Foundations of Artificial Intelligence at Northeastern University under the guidance of Dr. Sarita Singh.

<strong>Citation</strong>

If you reference this academic prototype, you may cite it as:

@software{healthbridge_ai_2026,
  author = {Gajawada, Preethi and Cherukuri, Madhu Babu},
  title = {HealthBridge AI: An Inclusive and Secure Conversational Assistant for Remote Patient Mobility Monitoring},
  year = {2026},
  url = {https://github.com/gajawp/HealthCareMonitoringMobilitySystem}
}

For questions, issues, or improvement proposals, open an issue in the repository.
