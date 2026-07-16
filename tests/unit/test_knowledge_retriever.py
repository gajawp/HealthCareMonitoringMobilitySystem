from pathlib import Path
import sys

APPLICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APPLICATION_DIR))

from app.chatbot.knowledge_retriever import ClinicalKnowledgeRetriever


def test_knee_extension_search_returns_results():
    retriever = ClinicalKnowledgeRetriever()
    results = retriever.search("How should I raise my leg for knee extension?")
    assert results
    assert any(result.get("exercise_id") == "seated_knee_extension" for result in results)


def test_empty_query_returns_no_results():
    retriever = ClinicalKnowledgeRetriever()
    assert retriever.search("   ") == []


def test_exercise_filter():
    retriever = ClinicalKnowledgeRetriever()
    results = retriever.search(
        "alignment and balance",
        exercise_id="hurdle_step",
    )
    assert all(result.get("exercise_id") == "hurdle_step" for result in results)
