from __future__ import annotations

import base64
import io
from typing import Optional

from gtts import gTTS


# gTTS-compatible language codes.
VOICE_LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Urdu": "ur",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
}


def get_voice_language_code(selected_language: str) -> str:
    """
    Return the speech language code for the dashboard language.

    Defaults to English when the selected language is unsupported.
    """

    return VOICE_LANGUAGE_CODES.get(selected_language, "en")


def text_to_speech_html(
    text: str,
    selected_language: str,
    autoplay: bool = False,
) -> Optional[str]:
    """
    Convert chatbot text into speech and return an HTML audio player.

    Audio is generated in memory and is not stored permanently.
    """

    if not text or not text.strip():
        return None

    language_code = get_voice_language_code(selected_language)

    try:
        audio_buffer = io.BytesIO()

        speech = gTTS(
            text=text,
            lang=language_code,
            slow=False,
        )
        speech.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        encoded_audio = base64.b64encode(
            audio_buffer.read()
        ).decode("utf-8")

        autoplay_attribute = "autoplay" if autoplay else ""

        return f"""
        <audio controls {autoplay_attribute} style="width: 100%;">
            <source
                src="data:audio/mp3;base64,{encoded_audio}"
                type="audio/mpeg"
            >
            Your browser does not support audio playback.
        </audio>
        """

    except Exception:
        return None