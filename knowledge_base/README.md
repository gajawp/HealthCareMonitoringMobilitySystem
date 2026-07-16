# Healthcare Mobility Clinical Knowledge Base

This folder contains structured, reviewable knowledge used by the Healthcare Mobility Assistant.

## Important limitation

The content is for a software prototype and research workflow. It is not a prescription, diagnosis, or patient-specific rehabilitation plan. All records begin with:

```json
"clinical_review_status": "pending"
```

A qualified reviewer should approve exercise content before deployment.

## Structure

- `exercises/`: exercise descriptions, phases, observable errors, general guidance, and safety limits.
- `metrics/`: dashboard metric and sensor definitions.
- `safety/`: chatbot boundaries and escalation rules.
- `sources/`: source provenance, intended use, and license notes.
- `generated/knowledge_chunks.json`: generated retrieval records.

## Build

From `Application/`:

```bash
python scripts/validate_knowledge_base.py
python scripts/build_knowledge_base.py
```

## Test retrieval

```bash
python -c "from chatbot.knowledge_retriever import ClinicalKnowledgeRetriever; print(ClinicalKnowledgeRetriever().search('explain knee extension'))"
```

## Run tests

```bash
python -m pip install pytest
python -m pytest tests/test_knowledge_retriever.py -v
```

## Updating content

1. Edit a source JSON file.
2. Keep clinical thresholds `null` unless they come from the configured care plan.
3. Update the source registry.
4. Run validation.
5. Rebuild `knowledge_chunks.json`.
6. Run tests.
7. Request clinical review before changing status to `approved`.

## Source-use principle

Research datasets can support movement modeling and terminology, but dataset classes and average values must not automatically become clinical targets.
