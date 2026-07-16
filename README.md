# HealthCareMobilityMonitoringSystem

A unified Streamlit healthcare mobility dashboard with an integrated,
retrieval-grounded AI assistant for patients, caregivers, and clinicians.

## Structure

```text
app/                 Dashboard and chatbot backend
knowledge_base/      Exercises, metrics, safety rules, sources, generated chunks
data/                Mobility and tremor sample data
scripts/             Knowledge-base, report, and MQTT utilities
tests/               Unit and integration tests
docs/references/     Research references
reports/             Generated reports and assets
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

For no-cost local fallback:

```env
USE_LLM=false
```

For OpenAI:

```env
OPENAI_API_KEY=your_new_key
OPENAI_MODEL=gpt-5-mini
USE_LLM=true
```

Never commit `.env`.

## Validate and test

```bash
python scripts/validate_knowledge_base.py
python scripts/build_knowledge_base.py
python -m pytest
```

## Run

```bash
python run_app.py
```

Or:

```bash
python -m streamlit run app/main.py
```

## Demo users

Password for all demo accounts: `pass123`

- Patient: `P1001`
- Caregiver: `C2001`
- Clinician: `D3001`

## Safety scope

The chatbot explains recorded mobility information and approved exercise
material. It does not diagnose conditions, prescribe medication, or change
a patient-specific care plan.
