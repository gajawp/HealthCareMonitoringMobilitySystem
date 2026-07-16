from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"


def test_chatbot_ui_syntax():
    ast.parse((APP / "chatbot_ui.py").read_text(encoding="utf-8"))


def test_phase3_app_syntax():
    ast.parse((APP / "main.py").read_text(encoding="utf-8"))


def test_phase3_markers_present():
    text = (APP / "main.py").read_text(encoding="utf-8")
    assert "from chatbot_ui import render_chatbot" in text
    assert '"AI Assistant"' in text
    assert 'role="Patient"' in text
    assert 'role="Caregiver"' in text
    assert 'role="Clinician"' in text
