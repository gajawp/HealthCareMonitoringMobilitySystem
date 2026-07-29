from __future__ import annotations

import base64
import html
import io
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

from app.chatbot.language_service import LanguageService
from app.chatbot.orchestrator import (
    HealthcareChatbotOrchestrator,
)


LANGUAGE_OPTIONS: dict[str, str] = {
    "Automatic - Auto detect": "auto",
    "English - English": "en",
    "Spanish - Español": "es",
    "French - Français": "fr",
    "German - Deutsch": "de",
    "Hindi - हिन्दी": "hi",
    "Telugu - తెలుగు": "te",
    "Tamil - தமிழ்": "ta",
    "Chinese - 中文": "zh",
    "Arabic - العربية": "ar",
}


VOICE_UI_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "heading": "Voice assistant",
        "start": "Start recording",
        "stop": "Stop recording",
        "processing": "Converting speech to text...",
        "heard": "I heard",
        "play": "Listen to response",
        "unavailable": (
            "Voice support requires gTTS, openai, and "
            "streamlit-mic-recorder."
        ),
        "error": "Voice processing failed.",
    },
    "es": {
        "heading": "Asistente de voz",
        "start": "Iniciar grabación",
        "stop": "Detener grabación",
        "processing": "Convirtiendo voz a texto...",
        "heard": "Escuché",
        "play": "Escuchar la respuesta",
        "unavailable": "El soporte de voz no está disponible.",
        "error": "Falló el procesamiento de voz.",
    },
    "fr": {
        "heading": "Assistant vocal",
        "start": "Démarrer l’enregistrement",
        "stop": "Arrêter l’enregistrement",
        "processing": "Conversion de la voix en texte...",
        "heard": "J’ai entendu",
        "play": "Écouter la réponse",
        "unavailable": "L’assistance vocale n’est pas disponible.",
        "error": "Le traitement vocal a échoué.",
    },
    "de": {
        "heading": "Sprachassistent",
        "start": "Aufnahme starten",
        "stop": "Aufnahme stoppen",
        "processing": "Sprache wird in Text umgewandelt...",
        "heard": "Ich habe gehört",
        "play": "Antwort anhören",
        "unavailable": "Sprachunterstützung ist nicht verfügbar.",
        "error": "Die Sprachverarbeitung ist fehlgeschlagen.",
    },
    "hi": {
        "heading": "वॉइस सहायक",
        "start": "रिकॉर्डिंग शुरू करें",
        "stop": "रिकॉर्डिंग रोकें",
        "processing": "आवाज़ को टेक्स्ट में बदला जा रहा है...",
        "heard": "मैंने सुना",
        "play": "उत्तर सुनें",
        "unavailable": "वॉइस सहायता उपलब्ध नहीं है।",
        "error": "वॉइस प्रोसेसिंग विफल रही।",
    },
    "te": {
        "heading": "వాయిస్ సహాయకుడు",
        "start": "రికార్డింగ్ ప్రారంభించండి",
        "stop": "రికార్డింగ్ ఆపండి",
        "processing": "మాటలను పాఠ్యంగా మారుస్తోంది...",
        "heard": "నేను విన్నది",
        "play": "సమాధానాన్ని వినండి",
        "unavailable": "వాయిస్ సహాయం అందుబాటులో లేదు.",
        "error": "వాయిస్ ప్రాసెసింగ్ విఫలమైంది.",
    },
    "ta": {
        "heading": "குரல் உதவியாளர்",
        "start": "பதிவைத் தொடங்கவும்",
        "stop": "பதிவை நிறுத்தவும்",
        "processing": "குரல் உரையாக மாற்றப்படுகிறது...",
        "heard": "நான் கேட்டது",
        "play": "பதிலைக் கேட்கவும்",
        "unavailable": "குரல் உதவி கிடைக்கவில்லை.",
        "error": "குரல் செயலாக்கம் தோல்வியடைந்தது.",
    },
    "zh": {
        "heading": "语音助手",
        "start": "开始录音",
        "stop": "停止录音",
        "processing": "正在将语音转换为文字...",
        "heard": "我听到的是",
        "play": "收听回复",
        "unavailable": "语音支持不可用。",
        "error": "语音处理失败。",
    },
    "ar": {
        "heading": "المساعد الصوتي",
        "start": "بدء التسجيل",
        "stop": "إيقاف التسجيل",
        "processing": "جارٍ تحويل الصوت إلى نص...",
        "heard": "سمعت",
        "play": "الاستماع إلى الرد",
        "unavailable": "الدعم الصوتي غير متاح.",
        "error": "فشلت معالجة الصوت.",
    },
}


TRANSCRIPTION_LANGUAGE_CODES: dict[str, str] = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "hi": "hi",
    "te": "te",
    "ta": "ta",
    "zh": "zh",
    "ar": "ar",
}


TTS_LANGUAGE_CODES: dict[str, str] = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "hi": "hi",
    "te": "te",
    "ta": "ta",
    "zh": "zh-CN",
    "ar": "ar",
}


EXERCISE_DISPLAY_NAMES: dict[str, dict[str, str]] = {
    "en": {
        "seated knee extension": "Seated Knee Extension",
        "sit-to-stand exercise": "Sit-to-Stand Exercise",
        "seated marching": "Seated Marching",
        "finger tapping drill": "Finger Tapping Drill",
        "hand open-close repetitions": "Hand Open-Close Repetitions",
        "resting tremor assessment": "Resting Tremor Assessment",
        "leg raise exercise": "Leg Raise Exercise",
    },
    "es": {
        "seated knee extension": "Extensión de rodilla sentado",
        "sit-to-stand exercise": "Ejercicio de sentarse y levantarse",
        "seated marching": "Marcha sentado",
        "finger tapping drill": "Ejercicio de golpeteo con los dedos",
        "hand open-close repetitions": "Repeticiones de abrir y cerrar la mano",
        "resting tremor assessment": "Evaluación del temblor en reposo",
        "leg raise exercise": "Ejercicio de elevación de pierna",
    },
    "fr": {
        "seated knee extension": "Extension du genou en position assise",
        "sit-to-stand exercise": "Exercice assis-debout",
        "seated marching": "Marche en position assise",
        "finger tapping drill": "Exercice de tapotement des doigts",
        "hand open-close repetitions": "Répétitions d’ouverture et de fermeture de la main",
        "resting tremor assessment": "Évaluation du tremblement au repos",
        "leg raise exercise": "Exercice de levée de jambe",
    },
    "de": {
        "seated knee extension": "Kniestreckung im Sitzen",
        "sit-to-stand exercise": "Aufstehübung",
        "seated marching": "Marschieren im Sitzen",
        "finger tapping drill": "Fingerklopfübung",
        "hand open-close repetitions": "Hand öffnen und schließen",
        "resting tremor assessment": "Beurteilung des Ruhetremors",
        "leg raise exercise": "Beinhebeübung",
    },
    "hi": {
        "seated knee extension": "बैठकर घुटना सीधा करना",
        "sit-to-stand exercise": "बैठने से खड़े होने का व्यायाम",
        "seated marching": "बैठकर मार्च करना",
        "finger tapping drill": "उंगली टैपिंग अभ्यास",
        "hand open-close repetitions": "हाथ खोलने और बंद करने की पुनरावृत्तियाँ",
        "resting tremor assessment": "आराम की स्थिति में कंपकंपी का आकलन",
        "leg raise exercise": "पैर उठाने का व्यायाम",
    },
    "te": {
        "seated knee extension": "కూర్చుని మోకాలి పొడిగింపు",
        "sit-to-stand exercise": "కూర్చుని నిలబడే వ్యాయామం",
        "seated marching": "కూర్చుని మార్చింగ్",
        "finger tapping drill": "వేలితో ట్యాపింగ్ వ్యాయామం",
        "hand open-close repetitions": "చేతిని తెరవడం-మూయడం పునరావృతాలు",
        "resting tremor assessment": "విశ్రాంతి వణుకు అంచనా",
        "leg raise exercise": "కాలు ఎత్తే వ్యాయామం",
    },
    "ta": {
        "seated knee extension": "அமர்ந்த நிலையில் முழங்கால் நீட்டிப்பு",
        "sit-to-stand exercise": "அமர்ந்து எழும் பயிற்சி",
        "seated marching": "அமர்ந்த நிலையில் நடைப்பயிற்சி",
        "finger tapping drill": "விரல் தட்டும் பயிற்சி",
        "hand open-close repetitions": "கையைத் திறந்து மூடும் மறுபடிகள்",
        "resting tremor assessment": "ஓய்வு நடுக்க மதிப்பீடு",
        "leg raise exercise": "கால் உயர்த்தும் பயிற்சி",
    },
    "zh": {
        "seated knee extension": "坐姿膝关节伸展",
        "sit-to-stand exercise": "坐站练习",
        "seated marching": "坐姿踏步",
        "finger tapping drill": "手指敲击练习",
        "hand open-close repetitions": "手掌张合重复练习",
        "resting tremor assessment": "静止性震颤评估",
        "leg raise exercise": "抬腿练习",
    },
    "ar": {
        "seated knee extension": "تمديد الركبة أثناء الجلوس",
        "sit-to-stand exercise": "تمرين الجلوس والوقوف",
        "seated marching": "المشي في وضع الجلوس",
        "finger tapping drill": "تمرين نقر الأصابع",
        "hand open-close repetitions": "تكرار فتح وإغلاق اليد",
        "resting tremor assessment": "تقييم الرعاش أثناء الراحة",
        "leg raise exercise": "تمرين رفع الساق",
    },
}


UI_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "title": "💬 AI Healthcare Mobility Assistant",
        "notice": (
            "This assistant explains recorded dashboard information and "
            "approved exercise material. It does not diagnose conditions "
            "or change a care plan."
        ),
        "language": "Chat language",
        "signed_in": "Signed in as",
        "role": "Role",
        "selected_patient": "Selected patient",
        "suggested_questions": "Suggested questions",
        "exercise_selector": "Choose an exercise for guidance",
        "chat_input": (
            "Ask about sessions, alerts, mobility metrics, or exercises"
        ),
        "reviewing": (
            "Reviewing mobility data and approved guidance..."
        ),
        "sources": "Knowledge sources used",
        "starting_position": "Starting position",
        "how_to": "How to perform it",
        "video": "Demonstration video",
        "open_video": "Open demonstration video",
        "clear": "Clear conversation",
        "session_storage": (
            "Conversation history is stored only in the current "
            "Streamlit session."
        ),
        "technical_details": "Technical response details",
        "escalation": (
            "This response includes a care-team or urgent escalation."
        ),
        "error": (
            "The mobility assistant could not complete this request. "
            "Please review the application terminal for the technical error."
        ),
        "no_exercises": (
            "No exercises are currently configured for this patient profile."
        ),
        "suggested_exercises_heading": "Suggested Exercises",
    },
    "es": {
        "title": "💬 Asistente de movilidad médica con IA",
        "notice": (
            "Este asistente explica la información registrada del panel y "
            "el material de ejercicios aprobado. No diagnostica afecciones "
            "ni modifica un plan de atención."
        ),
        "language": "Idioma del chat",
        "signed_in": "Sesión iniciada como",
        "role": "Rol",
        "selected_patient": "Paciente seleccionado",
        "suggested_questions": "Preguntas sugeridas",
        "exercise_selector": "Elija un ejercicio para recibir orientación",
        "chat_input": (
            "Pregunte sobre sesiones, alertas, métricas de movilidad "
            "o ejercicios"
        ),
        "reviewing": "Revisando datos de movilidad y orientación aprobada...",
        "sources": "Fuentes de conocimiento utilizadas",
        "starting_position": "Posición inicial",
        "how_to": "Cómo realizarlo",
        "video": "Video de demostración",
        "open_video": "Abrir video de demostración",
        "clear": "Borrar conversación",
        "session_storage": (
            "El historial se guarda solo en la sesión actual de Streamlit."
        ),
        "technical_details": "Detalles técnicos de la respuesta",
        "escalation": (
            "Esta respuesta incluye una derivación al equipo de atención "
            "o una derivación urgente."
        ),
        "error": "El asistente de movilidad no pudo completar la solicitud.",
        "no_exercises": (
            "No hay ejercicios configurados para este perfil de paciente."
        ),
        "suggested_exercises_heading": "Ejercicios sugeridos",
    },
    "hi": {
        "title": "💬 एआई स्वास्थ्य गतिशीलता सहायक",
        "notice": (
            "यह सहायक रिकॉर्ड किए गए डैशबोर्ड डेटा और स्वीकृत व्यायाम "
            "सामग्री को समझाता है। यह निदान नहीं करता और देखभाल योजना "
            "नहीं बदलता।"
        ),
        "language": "चैट भाषा",
        "signed_in": "साइन इन उपयोगकर्ता",
        "role": "भूमिका",
        "selected_patient": "चयनित रोगी",
        "suggested_questions": "सुझाए गए प्रश्न",
        "exercise_selector": "मार्गदर्शन के लिए व्यायाम चुनें",
        "chat_input": (
            "सत्र, चेतावनी, गतिशीलता मेट्रिक या व्यायाम के बारे में पूछें"
        ),
        "reviewing": "गतिशीलता डेटा और स्वीकृत मार्गदर्शन की समीक्षा हो रही है...",
        "sources": "उपयोग किए गए ज्ञान स्रोत",
        "starting_position": "प्रारंभिक स्थिति",
        "how_to": "इसे कैसे करें",
        "video": "प्रदर्शन वीडियो",
        "open_video": "प्रदर्शन वीडियो खोलें",
        "clear": "बातचीत साफ़ करें",
        "session_storage": (
            "बातचीत का इतिहास केवल वर्तमान Streamlit सत्र में रखा जाता है।"
        ),
        "technical_details": "तकनीकी प्रतिक्रिया विवरण",
        "escalation": (
            "इस प्रतिक्रिया में देखभाल टीम या तत्काल सहायता की सलाह शामिल है।"
        ),
        "error": "गतिशीलता सहायक अनुरोध पूरा नहीं कर सका।",
        "no_exercises": (
            "इस रोगी प्रोफ़ाइल के लिए कोई व्यायाम कॉन्फ़िगर नहीं है।"
        ),
        "suggested_exercises_heading": "सुझाए गए व्यायाम",
    },
    "te": {
        "title": "💬 ఏఐ ఆరోగ్య మొబిలిటీ సహాయకుడు",
        "notice": (
            "ఈ సహాయకుడు రికార్డ్ చేసిన డ్యాష్‌బోర్డ్ సమాచారం మరియు "
            "ఆమోదించబడిన వ్యాయామ సమాచారాన్ని వివరిస్తాడు. ఇది వ్యాధిని "
            "నిర్ధారించదు లేదా చికిత్స ప్రణాళికను మార్చదు."
        ),
        "language": "చాట్ భాష",
        "signed_in": "సైన్ ఇన్ చేసిన వినియోగదారు",
        "role": "పాత్ర",
        "selected_patient": "ఎంచుకున్న రోగి",
        "suggested_questions": "సూచించిన ప్రశ్నలు",
        "exercise_selector": "మార్గదర్శకత్వం కోసం వ్యాయామాన్ని ఎంచుకోండి",
        "chat_input": (
            "సెషన్లు, హెచ్చరికలు, మొబిలిటీ కొలతలు లేదా వ్యాయామాల గురించి అడగండి"
        ),
        "reviewing": "మొబిలిటీ డేటా మరియు ఆమోదిత మార్గదర్శకత్వాన్ని పరిశీలిస్తోంది...",
        "sources": "ఉపయోగించిన జ్ఞాన వనరులు",
        "starting_position": "ప్రారంభ స్థానం",
        "how_to": "ఎలా చేయాలి",
        "video": "ప్రదర్శన వీడియో",
        "open_video": "ప్రదర్శన వీడియోను తెరవండి",
        "clear": "సంభాషణను తొలగించండి",
        "session_storage": (
            "సంభాషణ చరిత్ర ప్రస్తుత Streamlit సెషన్‌లో మాత్రమే నిల్వ ఉంటుంది."
        ),
        "technical_details": "సాంకేతిక స్పందన వివరాలు",
        "escalation": (
            "ఈ స్పందనలో సంరక్షణ బృందం లేదా అత్యవసర సహాయం సూచన ఉంది."
        ),
        "error": "మొబిలిటీ సహాయకుడు అభ్యర్థనను పూర్తి చేయలేకపోయాడు.",
        "no_exercises": (
            "ఈ రోగి ప్రొఫైల్‌కు వ్యాయామాలు కాన్ఫిగర్ చేయబడలేదు."
        ),
        "suggested_exercises_heading": "సూచించిన వ్యాయామాలు",
    },

    "fr": {
        "title": "💬 Assistant de mobilité médicale IA",
        "notice": (
            "Cet assistant explique les informations enregistrées dans le "
            "tableau de bord et les exercices approuvés. Il ne diagnostique "
            "pas de pathologie et ne modifie pas le plan de soins."
        ),
        "language": "Langue du chat",
        "signed_in": "Connecté en tant que",
        "role": "Rôle",
        "selected_patient": "Patient sélectionné",
        "suggested_questions": "Questions suggérées",
        "exercise_selector": "Choisissez un exercice pour obtenir des conseils",
        "chat_input": (
            "Posez une question sur les séances, les alertes, les mesures "
            "de mobilité ou les exercices"
        ),
        "reviewing": (
            "Analyse des données de mobilité et des conseils approuvés..."
        ),
        "sources": "Sources de connaissances utilisées",
        "starting_position": "Position de départ",
        "how_to": "Comment effectuer l’exercice",
        "video": "Vidéo de démonstration",
        "open_video": "Ouvrir la vidéo de démonstration",
        "clear": "Effacer la conversation",
        "session_storage": (
            "L’historique de la conversation est conservé uniquement dans "
            "la session Streamlit actuelle."
        ),
        "technical_details": "Détails techniques de la réponse",
        "escalation": (
            "Cette réponse recommande de contacter l’équipe soignante "
            "ou d’obtenir une aide urgente."
        ),
        "error": (
            "L’assistant de mobilité n’a pas pu traiter cette demande."
        ),
        "no_exercises": (
            "Aucun exercice n’est actuellement configuré pour ce profil."
        ),
        "suggested_exercises_heading": "Exercices suggérés",
    },
    "de": {
        "title": "💬 KI-Assistent für medizinische Mobilität",
        "notice": (
            "Dieser Assistent erklärt aufgezeichnete Dashboard-Informationen "
            "und freigegebene Übungsmaterialien. Er stellt keine Diagnose "
            "und ändert keinen Behandlungsplan."
        ),
        "language": "Chatsprache",
        "signed_in": "Angemeldet als",
        "role": "Rolle",
        "selected_patient": "Ausgewählter Patient",
        "suggested_questions": "Vorgeschlagene Fragen",
        "exercise_selector": "Übung für die Anleitung auswählen",
        "chat_input": (
            "Fragen Sie nach Sitzungen, Warnungen, Mobilitätswerten "
            "oder Übungen"
        ),
        "reviewing": (
            "Mobilitätsdaten und freigegebene Anleitungen werden geprüft..."
        ),
        "sources": "Verwendete Wissensquellen",
        "starting_position": "Ausgangsposition",
        "how_to": "Durchführung",
        "video": "Demonstrationsvideo",
        "open_video": "Demonstrationsvideo öffnen",
        "clear": "Unterhaltung löschen",
        "session_storage": (
            "Der Gesprächsverlauf wird nur in der aktuellen "
            "Streamlit-Sitzung gespeichert."
        ),
        "technical_details": "Technische Antwortdetails",
        "escalation": (
            "Diese Antwort empfiehlt die Kontaktaufnahme mit dem "
            "Behandlungsteam oder dringende Hilfe."
        ),
        "error": (
            "Der Mobilitätsassistent konnte diese Anfrage nicht bearbeiten."
        ),
        "no_exercises": (
            "Für dieses Patientenprofil sind derzeit keine Übungen konfiguriert."
        ),
        "suggested_exercises_heading": "Vorgeschlagene Übungen",
    },
    "ta": {
        "title": "💬 ஏஐ சுகாதார இயக்க உதவியாளர்",
        "notice": (
            "இந்த உதவியாளர் பதிவு செய்யப்பட்ட டாஷ்போர்டு தகவல்களையும் "
            "அங்கீகரிக்கப்பட்ட உடற்பயிற்சி தகவல்களையும் விளக்குகிறது. "
            "இது நோய்களை கண்டறியாது அல்லது சிகிச்சைத் திட்டத்தை மாற்றாது."
        ),
        "language": "அரட்டை மொழி",
        "signed_in": "உள்நுழைந்தவர்",
        "role": "பங்கு",
        "selected_patient": "தேர்ந்தெடுக்கப்பட்ட நோயாளர்",
        "suggested_questions": "பரிந்துரைக்கப்பட்ட கேள்விகள்",
        "exercise_selector": (
            "வழிகாட்டலுக்கான உடற்பயிற்சியைத் தேர்ந்தெடுக்கவும்"
        ),
        "chat_input": (
            "அமர்வுகள், எச்சரிக்கைகள், இயக்க அளவீடுகள் அல்லது "
            "உடற்பயிற்சிகள் குறித்து கேளுங்கள்"
        ),
        "reviewing": (
            "இயக்கத் தரவுகளும் அங்கீகரிக்கப்பட்ட வழிகாட்டலும் "
            "மதிப்பாய்வு செய்யப்படுகின்றன..."
        ),
        "sources": "பயன்படுத்தப்பட்ட அறிவு ஆதாரங்கள்",
        "starting_position": "தொடக்க நிலை",
        "how_to": "எவ்வாறு செய்வது",
        "video": "செயல்விளக்க காணொளி",
        "open_video": "செயல்விளக்க காணொளியைத் திறக்கவும்",
        "clear": "உரையாடலை அழிக்கவும்",
        "session_storage": (
            "உரையாடல் வரலாறு தற்போதைய Streamlit அமர்வில் மட்டுமே "
            "சேமிக்கப்படும்."
        ),
        "technical_details": "தொழில்நுட்ப பதில் விவரங்கள்",
        "escalation": (
            "இந்த பதிலில் பராமரிப்பு குழு அல்லது அவசர உதவி பரிந்துரை உள்ளது."
        ),
        "error": (
            "இயக்க உதவியாளர் இந்த கோரிக்கையை முடிக்க முடியவில்லை."
        ),
        "no_exercises": (
            "இந்த நோயாளர் சுயவிவரத்திற்கு உடற்பயிற்சிகள் அமைக்கப்படவில்லை."
        ),
        "suggested_exercises_heading": (
            "பரிந்துரைக்கப்பட்ட உடற்பயிற்சிகள்"
        ),
    },
    "zh": {
        "title": "💬 AI 医疗行动能力助手",
        "notice": (
            "此助手用于解释已记录的仪表板信息和已批准的练习资料。"
            "它不会诊断疾病，也不会更改护理计划。"
        ),
        "language": "聊天语言",
        "signed_in": "当前登录用户",
        "role": "角色",
        "selected_patient": "所选患者",
        "suggested_questions": "推荐问题",
        "exercise_selector": "选择一项练习以查看指导",
        "chat_input": "询问训练记录、警报、行动能力指标或练习",
        "reviewing": "正在查看行动能力数据和已批准的指导...",
        "sources": "使用的知识来源",
        "starting_position": "起始姿势",
        "how_to": "练习方法",
        "video": "演示视频",
        "open_video": "打开演示视频",
        "clear": "清除对话",
        "session_storage": (
            "对话记录仅保存在当前 Streamlit 会话中。"
        ),
        "technical_details": "技术响应详情",
        "escalation": (
            "此回复包含联系护理团队或寻求紧急帮助的建议。"
        ),
        "error": "行动能力助手无法完成此请求。",
        "no_exercises": "此患者档案目前没有配置任何练习。",
        "suggested_exercises_heading": "推荐练习",
    },
    "ar": {
        "title": "💬 مساعد التنقل الصحي بالذكاء الاصطناعي",
        "notice": (
            "يشرح هذا المساعد معلومات لوحة المتابعة المسجلة ومواد "
            "التمارين المعتمدة. ولا يشخّص الحالات ولا يغيّر خطة الرعاية."
        ),
        "language": "لغة المحادثة",
        "signed_in": "تم تسجيل الدخول باسم",
        "role": "الدور",
        "selected_patient": "المريض المحدد",
        "suggested_questions": "الأسئلة المقترحة",
        "exercise_selector": "اختر تمرينًا للحصول على الإرشادات",
        "chat_input": (
            "اسأل عن الجلسات أو التنبيهات أو مقاييس الحركة أو التمارين"
        ),
        "reviewing": (
            "تتم مراجعة بيانات الحركة والإرشادات المعتمدة..."
        ),
        "sources": "مصادر المعرفة المستخدمة",
        "starting_position": "وضعية البداية",
        "how_to": "كيفية أداء التمرين",
        "video": "فيديو توضيحي",
        "open_video": "فتح الفيديو التوضيحي",
        "clear": "مسح المحادثة",
        "session_storage": (
            "يتم حفظ سجل المحادثة في جلسة Streamlit الحالية فقط."
        ),
        "technical_details": "تفاصيل الاستجابة التقنية",
        "escalation": (
            "تتضمن هذه الاستجابة توصية بالتواصل مع فريق الرعاية "
            "أو طلب مساعدة عاجلة."
        ),
        "error": "تعذر على مساعد الحركة إكمال هذا الطلب.",
        "no_exercises": (
            "لا توجد تمارين مهيأة حاليًا لملف هذا المريض."
        ),
        "suggested_exercises_heading": "التمارين المقترحة",
    },
}


def _voice_labels(language_code: str) -> dict[str, str]:
    return VOICE_UI_TEXT.get(language_code, VOICE_UI_TEXT["en"])


def _transcribe_audio(
    audio_bytes: bytes,
    language_code: str,
) -> str:
    """
    Transcribe recorded audio.

    The transcription API is allowed to detect the spoken language
    automatically. We intentionally do not pass the selected dashboard
    language as the API ``language`` parameter because codes such as
    Telugu (``te``) are rejected by some transcription models.
    """

    del language_code

    if OpenAI is None:
        raise RuntimeError(
            "The openai package is not installed."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(api_key=api_key)
    temporary_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temporary_file:
            temporary_file.write(audio_bytes)
            temporary_path = temporary_file.name

        with open(temporary_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )

        return transcription.text.strip()

    finally:
        if temporary_path and os.path.exists(
            temporary_path
        ):
            os.remove(temporary_path)


def _generate_speech(
    text: str,
    language_code: str,
) -> bytes | None:
    if not text.strip() or gTTS is None:
        return None

    audio_buffer = io.BytesIO()

    speech = gTTS(
        text=text,
        lang=TTS_LANGUAGE_CODES.get(
            language_code,
            "en",
        ),
        slow=False,
    )
    speech.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return audio_buffer.read()


def get_chatbot() -> HealthcareChatbotOrchestrator:
    return HealthcareChatbotOrchestrator()


def get_language_service() -> LanguageService:
    return LanguageService()


def _history_key(
    user_id: str,
    patient_id: str,
    language_code: str,
) -> str:
    return (
        f"mobility_chat_history::{user_id}::{patient_id}::"
        f"{language_code}"
    )


def _labels(language_code: str) -> dict[str, str]:
    return UI_TEXT.get(language_code, UI_TEXT["en"])


def _localized_text(
    *,
    language_service: LanguageService,
    text: str,
    language_code: str,
) -> str:
    if language_code == "en":
        return text

    return language_service.translate_from_english(
        text,
        language_code,
    )


def _initial_message(
    user_name: str,
    role: str,
    language_code: str,
    language_service: LanguageService,
) -> dict[str, str]:
    role_intro = {
        "Patient": (
            "I can summarize your recorded sessions, explain alerts and "
            "mobility metrics, and show your suggested exercises."
        ),
        "Caregiver": (
            "I can summarize the selected patient's recorded sessions, "
            "explain alerts, and retrieve approved exercise guidance."
        ),
        "Clinician": (
            "I can provide concise summaries of the selected patient's "
            "recorded sessions, flagged repetitions, mobility metrics, "
            "and source-backed exercise reference material."
        ),
    }

    english_message = (
        f"Hello {user_name}. "
        f"{role_intro.get(role, role_intro['Patient'])}"
    )

    return {
        "role": "assistant",
        "content": _localized_text(
            language_service=language_service,
            text=english_message,
            language_code=language_code,
        ),
    }


def _suggested_questions(
    role: str,
    selected_exercise: str | None,
) -> list[tuple[str, str]]:
    if role == "Clinician":
        prompts = [
            (
                "🏋️ Suggested exercises",
                "Show the suggested exercises for this patient.",
            ),
            (
                "📋 Session summary",
                "Summarize the latest recorded mobility session.",
            ),
            (
                "⚠️ Flagged repetitions",
                "Explain the flagged repetitions.",
            ),
            (
                "📐 Metric explanation",
                "What does jerk score mean?",
            ),
        ]

        if selected_exercise:
            prompts.append(
                (
                    f"🦵 Guidance: {selected_exercise}",
                    f"How should I perform {selected_exercise}?",
                )
            )

        return prompts

    if role == "Caregiver":
        prompts = [
            (
                "🏋️ Suggested exercises",
                "Show the suggested exercises for this patient.",
            ),
            (
                "📋 Latest session",
                "Summarize the latest mobility session.",
            ),
            (
                "⚠️ Explain alerts",
                "Why were repetitions flagged?",
            ),
            (
                "📈 Progress",
                "Is the patient improving over time?",
            ),
        ]

        if selected_exercise:
            prompts.append(
                (
                    f"🦵 Guidance: {selected_exercise}",
                    f"How should I perform {selected_exercise}?",
                )
            )

        return prompts

    return [
        (
            "🏋️ My suggested exercises",
            "Show my suggested exercises.",
        ),
        (
            "📋 My latest session",
            "Summarize my latest mobility session.",
        ),
        (
            "⚠️ Explain my alerts",
            "Why were my repetitions flagged?",
        ),
        (
            "📈 Am I improving?",
            "Am I improving over time?",
        ),
    ]


def _render_sources(
    sources: list[str],
    labels: dict[str, str],
) -> None:
    if not sources:
        return

    with st.expander(labels["sources"]):
        for source in sources:
            st.markdown(f"- `{source}`")


def _render_exercise_recommendations(
    exercise_data: dict[str, Any],
    labels: dict[str, str],
    *,
    compact: bool = False,
) -> None:
    profile_name = exercise_data.get(
        "profile_name",
        "Exercise Recommendations",
    )
    profile_description = exercise_data.get(
        "profile_description",
        "",
    )
    exercises = exercise_data.get("exercises", [])

    heading = "###" if compact else "##"
    st.markdown(
        f"{heading} {labels['suggested_exercises_heading']}: "
        f"{profile_name}"
    )

    if profile_description:
        st.write(profile_description)

    if not exercises:
        st.info(labels["no_exercises"])
        return

    for exercise in exercises:
        exercise_name = exercise.get(
            "exercise_name",
            "Exercise",
        )

        with st.container(border=True):
            st.markdown(
                f"{'####' if compact else '###'} {exercise_name}"
            )

            media = exercise.get("media", {})
            images = media.get("images", [])

            valid_images = [
                image
                for image in images
                if image.get("resolved_image_path")
                and Path(
                    image["resolved_image_path"]
                ).exists()
            ]

            if valid_images:
                column_count = min(len(valid_images), 2)
                columns = st.columns(column_count)

                for index, image in enumerate(valid_images):
                    with columns[index % column_count]:
                        image_path = str(
                            image["resolved_image_path"]
                        )
                        caption = str(
                            image.get("caption", "")
                        )
                        alt_text = str(
                            image.get(
                                "alt_text",
                                caption
                                or "Exercise demonstration",
                            )
                        )

                        try:
                            encoded_image = base64.b64encode(
                                Path(image_path).read_bytes()
                            ).decode("utf-8")
                        except OSError:
                            continue

                        suffix = Path(image_path).suffix.lower()

                        mime_type = {
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".png": "image/png",
                            ".webp": "image/webp",
                        }.get(suffix, "image/jpeg")

                        safe_caption = html.escape(caption)
                        safe_alt_text = html.escape(alt_text)

                        image_height = "210px" if compact else "360px"

                        st.markdown(
                            f"""
                            <div style="
                                width: 100%;
                                height: {image_height};
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                overflow: hidden;
                                border-radius: 8px;
                                background-color: #ffffff;
                            ">
                                <img
                                    src="data:{mime_type};base64,{encoded_image}"
                                    alt="{safe_alt_text}"
                                    style="
                                        width: 100%;
                                        height: 100%;
                                        object-fit: contain;
                                        object-position: center;
                                    "
                                />
                            </div>
                            <div style="
                                min-height: 42px;
                                text-align: center;
                                margin-top: 8px;
                                margin-bottom: 12px;
                                font-size: 0.95rem;
                                line-height: 1.4;
                            ">
                                {safe_caption}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            starting_position = exercise.get(
                "starting_position",
                [],
            )
            if starting_position:
                st.markdown(
                    f"**{labels['starting_position']}**"
                )
                for item in starting_position:
                    st.markdown(f"- {item}")

            guidance = exercise.get(
                "patient_friendly_guidance",
                [],
            )
            if guidance:
                st.markdown(f"**{labels['how_to']}**")
                for step_number, step in enumerate(
                    guidance,
                    start=1,
                ):
                    st.markdown(f"{step_number}. {step}")

            video_url = media.get("youtube_url")
            if video_url:
                st.markdown(f"**{labels['video']}**")
                try:
                    st.video(video_url)
                except Exception:
                    st.link_button(
                        labels["open_video"],
                        video_url,
                    )

            safety_notes = exercise.get(
                "safety_notes",
                [],
            )
            if safety_notes:
                st.warning(
                    "\n".join(
                        f"• {note}"
                        for note in safety_notes
                    )
                )

    safety_message = exercise_data.get(
        "global_safety_message",
        "",
    )
    if safety_message:
        st.info(safety_message)


def _render_chatbot_content(
    *,
    user_id: str,
    user_name: str,
    role: str,
    patient_id: str,
    patient_condition: str | None = None,
    authorized_patient_ids: list[str] | None = None,
    current_page: str = "AI Assistant",
    selected_session_id: str | None = None,
    selected_exercise_id: str | None = None,
    selected_date_range: str | None = None,
    visible_metrics: dict[str, Any] | None = None,
    active_alert: dict[str, Any] | None = None,
    compact: bool = False,
) -> None:
    language_service = get_language_service()

    language_key = f"chat_language::{user_id}::{patient_id}"

    if compact:
        with st.expander("🌐 Language and settings", expanded=False):
            language_name = st.selectbox(
                "Chat language / चैट भाषा / చాట్ భాష",
                options=list(LANGUAGE_OPTIONS.keys()),
                index=0,
                key=language_key,
            )
    else:
        language_name = st.selectbox(
            "Chat language / चैट भाषा / చాట్ భాష",
            options=list(LANGUAGE_OPTIONS.keys()),
            index=0,
            key=language_key,
        )
    selected_language = LANGUAGE_OPTIONS[language_name]

    language_state_key = (
        f"resolved_chat_language::{user_id}::{patient_id}"
    )
    if selected_language == "auto":
        resolved_language = st.session_state.get(
            language_state_key,
            "en",
        )
    else:
        resolved_language = selected_language
        st.session_state[
            language_state_key
        ] = selected_language

    ui_language = resolved_language
    labels = _labels(ui_language)

    if compact:
        st.caption(labels["notice"])
    else:
        st.title(labels["title"])
        st.info(labels["notice"])

        st.caption(
            f"{labels['signed_in']} **{user_name}** · "
            f"{labels['role']}: **{role}** · "
            f"{labels['selected_patient']}: **{patient_id}**"
        )

    history_language_key = (
        selected_language
        if selected_language == "auto"
        else ui_language
    )

    key = _history_key(
        user_id,
        patient_id,
        history_language_key,
    )

    if key not in st.session_state:
        st.session_state[key] = [
            _initial_message(
                user_name,
                role,
                ui_language,
                language_service,
            )
        ]

    if compact:
        st.markdown(f"**{labels['suggested_questions']}**")
    else:
        st.markdown(f"### {labels['suggested_questions']}")

    selected_exercise: str | None = None

    if role in {"Caregiver", "Clinician"}:
        exercise_options = [
            "seated knee extension",
            "sit-to-stand exercise",
            "seated marching",
            "finger tapping drill",
            "hand open-close repetitions",
            "resting tremor assessment",
            "leg raise exercise",
        ]

        default_exercise = (
            selected_exercise_id.strip().replace("_", " ")
            if isinstance(selected_exercise_id, str)
            and selected_exercise_id.strip()
            else exercise_options[0]
        )

        if default_exercise not in exercise_options:
            exercise_options.insert(0, default_exercise)

        display_names = EXERCISE_DISPLAY_NAMES.get(
            ui_language,
            EXERCISE_DISPLAY_NAMES["en"],
        )

        translated_options = [
            display_names.get(
                exercise_name,
                exercise_name,
            )
            for exercise_name in exercise_options
        ]

        default_display_name = display_names.get(
            default_exercise,
            default_exercise,
        )

        selected_display_name = st.selectbox(
            labels["exercise_selector"],
            options=translated_options,
            index=translated_options.index(
                default_display_name
            ),
            key=(
                f"exercise_guidance::{user_id}::"
                f"{patient_id}::{ui_language}"
            ),
        )

        display_to_internal = {
            display_names.get(
                exercise_name,
                exercise_name,
            ): exercise_name
            for exercise_name in exercise_options
        }

        selected_exercise = display_to_internal.get(
            selected_display_name,
            default_exercise,
        )

    suggestions = _suggested_questions(
        role,
        selected_exercise,
    )
    if compact:
        suggestions = suggestions[:2]

    columns = st.columns(1 if compact else 2)
    selected_prompt: str | None = None

    for index, (english_label, english_prompt) in enumerate(
        suggestions
    ):
        localized_label = _localized_text(
            language_service=language_service,
            text=english_label,
            language_code=ui_language,
        )

        with columns[index % len(columns)]:
            if st.button(
                localized_label,
                key=(
                    f"suggest::{user_id}::{patient_id}::"
                    f"{ui_language}::{index}"
                ),
                width="stretch",
            ):
                selected_prompt = english_prompt

    st.divider()

    for message in st.session_state[key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("audio")
            ):
                st.audio(
                    message["audio"],
                    format="audio/mp3",
                )

            if message.get("sources"):
                _render_sources(
                    message["sources"],
                    labels,
                )

            if (
                message.get("response_type")
                == "exercise_recommendations"
            ):
                _render_exercise_recommendations(
                    message.get("exercise_data", {}),
                    labels,
                    compact=compact,
                )

    voice_labels = _voice_labels(ui_language)
    voice_prompt: str | None = None

    recorded_audio = None

    if compact:
        input_col, voice_col, send_col = st.columns(
            [7, 1, 1],
            vertical_alignment="bottom",
            gap="small",
        )

        with input_col:
            typed_prompt = st.text_input(
                labels["chat_input"],
                key=(
                    f"floating_chat_text::{user_id}::"
                    f"{patient_id}::{ui_language}"
                ),
                label_visibility="collapsed",
                placeholder=labels["chat_input"],
            )

        with voice_col:
            if mic_recorder is None:
                st.button(
                    "🎙️",
                    key=(
                        f"voice_unavailable::{user_id}::"
                        f"{patient_id}::{ui_language}"
                    ),
                    help=voice_labels["unavailable"],
                    disabled=True,
                    use_container_width=True,
                )
            else:
                recorded_audio = mic_recorder(
                    start_prompt="🎙️",
                    stop_prompt="⏹️",
                    just_once=True,
                    use_container_width=True,
                    key=(
                        f"voice_recorder::{user_id}::"
                        f"{patient_id}::{ui_language}"
                    ),
                )

        with send_col:
            send_clicked = st.button(
                "➤",
                key=(
                    f"floating_chat_send::{user_id}::"
                    f"{patient_id}::{ui_language}"
                ),
                help="Send message",
                use_container_width=True,
            )

        if not send_clicked:
            typed_prompt = None

    else:
        st.markdown(f"### 🎙️ {voice_labels['heading']}")

        if mic_recorder is None:
            st.info(voice_labels["unavailable"])
        else:
            recorded_audio = mic_recorder(
                start_prompt=voice_labels["start"],
                stop_prompt=voice_labels["stop"],
                just_once=True,
                use_container_width=True,
                key=(
                    f"voice_recorder::{user_id}::"
                    f"{patient_id}::{ui_language}"
                ),
            )

        typed_prompt = st.chat_input(
            labels["chat_input"]
        )

    if (
        recorded_audio
        and recorded_audio.get("bytes")
    ):
        audio_signature = hash(
            recorded_audio["bytes"]
        )
        processed_audio_key = (
            f"processed_voice::{user_id}::"
            f"{patient_id}::{ui_language}"
        )

        if (
            st.session_state.get(
                processed_audio_key
            )
            != audio_signature
        ):
            try:
                with st.spinner(
                    voice_labels["processing"]
                ):
                    voice_prompt = _transcribe_audio(
                        recorded_audio["bytes"],
                        ui_language,
                    )

                st.session_state[
                    processed_audio_key
                ] = audio_signature
                st.session_state[
                    f"voice_text::{user_id}::"
                    f"{patient_id}::{ui_language}"
                ] = voice_prompt

            except Exception as exc:
                st.error(
                    f"{voice_labels['error']} {exc}"
                )
        else:
            voice_prompt = st.session_state.get(
                f"voice_text::{user_id}::"
                f"{patient_id}::{ui_language}"
            )

        if voice_prompt:
            st.success(
                f"{voice_labels['heard']}: "
                f"{voice_prompt}"
            )

    displayed_prompt = typed_prompt or voice_prompt
    english_prompt = selected_prompt

    if displayed_prompt:
        if selected_language == "auto":
            detected_language = (
                language_service.detect_language(
                    displayed_prompt
                )
            )
            st.session_state[
                language_state_key
            ] = detected_language
            resolved_language = detected_language
        else:
            resolved_language = selected_language

        english_prompt = (
            language_service.translate_to_english(
                displayed_prompt,
                resolved_language,
            )
        )

    if english_prompt:
        user_message = (
            displayed_prompt
            if displayed_prompt
            else _localized_text(
                language_service=language_service,
                text=english_prompt,
                language_code=resolved_language,
            )
        )

        st.session_state[key].append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            try:
                bot = get_chatbot()

                with st.spinner(labels["reviewing"]):
                    response = bot.answer(
                        question=english_prompt,
                        user_id=user_id,
                        user_name=user_name,
                        role=role,
                        patient_id=patient_id,
                        patient_condition=patient_condition,
                        authorized_patient_ids=authorized_patient_ids,
                        current_page=current_page,
                        selected_session_id=selected_session_id,
                        selected_exercise_id=selected_exercise_id,
                        selected_date_range=selected_date_range,
                        visible_metrics=visible_metrics,
                        active_alert=active_alert,
                    )

                localized_answer = (
                    language_service.translate_from_english(
                        response.answer,
                        resolved_language,
                    )
                )

                localized_exercise_data = (
                    language_service.translate_exercise_data(
                        response.exercise_data,
                        resolved_language,
                    )
                )

                response_audio = None
                try:
                    response_audio = _generate_speech(
                        localized_answer,
                        resolved_language,
                    )
                except Exception:
                    response_audio = None

                st.markdown(localized_answer)
                if response_audio:
                    st.audio(
                        response_audio,
                        format="audio/mp3",
                    )
                _render_sources(
                    response.sources,
                    labels,
                )

                if (
                    response.response_type
                    == "exercise_recommendations"
                    and localized_exercise_data
                ):
                    _render_exercise_recommendations(
                        localized_exercise_data,
                        labels,
                        compact=compact,
                    )

                if response.escalation_required:
                    st.warning(labels["escalation"])

                if role == "Clinician":
                    with st.expander(
                        labels["technical_details"]
                    ):
                        st.json(
                            {
                                "intent": response.intent,
                                "intent_confidence": (
                                    response.intent_confidence
                                ),
                                "provider": response.provider,
                                "model": response.model,
                                "used_llm": response.used_llm,
                                "safety_action": (
                                    response.safety_action
                                ),
                                "language": resolved_language,
                                "metadata": response.metadata,
                            }
                        )

                assistant_message = {
                    "role": "assistant",
                    "content": localized_answer,
                    "sources": response.sources,
                    "response_type": response.response_type,
                    "exercise_data": localized_exercise_data,
                    "audio": response_audio,
                }

            except Exception as exc:
                st.error(labels["error"])
                st.exception(exc)

                assistant_message = {
                    "role": "assistant",
                    "content": labels["error"],
                    "sources": [],
                }

        st.session_state[key].append(
            assistant_message
        )
        st.rerun()

    if compact:
        with st.expander("Chat options", expanded=False):
            if st.button(
                labels["clear"],
                key=(
                    f"clear::{user_id}::{patient_id}::"
                    f"{ui_language}"
                ),
                use_container_width=True,
            ):
                st.session_state[key] = [
                    _initial_message(
                        user_name,
                        role,
                        ui_language,
                        language_service,
                    )
                ]
                st.rerun()
            st.caption(labels["session_storage"])
    else:
        st.divider()
        clear_col, status_col = st.columns([1, 3])

        with clear_col:
            if st.button(
                labels["clear"],
                key=(
                    f"clear::{user_id}::{patient_id}::"
                    f"{ui_language}"
                ),
            ):
                st.session_state[key] = [
                    _initial_message(
                        user_name,
                        role,
                        ui_language,
                        language_service,
                    )
                ]
                st.rerun()

        with status_col:
            st.caption(labels["session_storage"])



def _enable_all_side_chat_resize() -> None:
    """Enable resizing from every edge and corner of the chat panel."""

    components.html(
        """
        <script>
        (() => {
            const parentWindow = window.parent;
            const parentDocument = parentWindow.document;
            const selector = ".st-key-healthbridge_chat_panel";
            const storageWidth = "healthbridge-chat-width";
            const storageHeight = "healthbridge-chat-height";

            function initializeResize() {
                const panel = parentDocument.querySelector(selector);

                if (!panel || panel.dataset.allSideResizeReady === "true") {
                    return;
                }

                panel.dataset.allSideResizeReady = "true";
                panel.style.boxSizing = "border-box";
                panel.style.position = "fixed";

                const savedWidth = parentWindow.localStorage.getItem(storageWidth);
                const savedHeight = parentWindow.localStorage.getItem(storageHeight);

                if (savedWidth) {
                    panel.style.setProperty("width", savedWidth, "important");
                }
                if (savedHeight) {
                    panel.style.setProperty("height", savedHeight, "important");
                }

                const handles = {
                    top: {
                        cursor: "ns-resize", top: "-5px", left: "12px",
                        right: "12px", height: "10px"
                    },
                    bottom: {
                        cursor: "ns-resize", bottom: "-5px", left: "12px",
                        right: "12px", height: "10px"
                    },
                    left: {
                        cursor: "ew-resize", left: "-5px", top: "12px",
                        bottom: "12px", width: "10px"
                    },
                    right: {
                        cursor: "ew-resize", right: "-5px", top: "12px",
                        bottom: "12px", width: "10px"
                    },
                    topLeft: {
                        cursor: "nwse-resize", top: "-7px", left: "-7px",
                        width: "16px", height: "16px"
                    },
                    topRight: {
                        cursor: "nesw-resize", top: "-7px", right: "-7px",
                        width: "16px", height: "16px"
                    },
                    bottomLeft: {
                        cursor: "nesw-resize", bottom: "-7px", left: "-7px",
                        width: "16px", height: "16px"
                    },
                    bottomRight: {
                        cursor: "nwse-resize", bottom: "-7px", right: "-7px",
                        width: "18px", height: "18px"
                    }
                };

                Object.entries(handles).forEach(([direction, styles]) => {
                    const handle = parentDocument.createElement("div");
                    handle.className = `healthbridge-resize-handle ${direction}`;
                    handle.dataset.resizeDirection = direction;
                    handle.style.position = "absolute";
                    handle.style.zIndex = "1000002";
                    handle.style.touchAction = "none";
                    handle.style.userSelect = "none";
                    Object.assign(handle.style, styles);

                    if (direction === "bottomRight") {
                        handle.style.borderRight =
                            "3px solid rgba(190,190,190,0.8)";
                        handle.style.borderBottom =
                            "3px solid rgba(190,190,190,0.8)";
                        handle.style.borderRadius = "0 0 5px 0";
                    }

                    panel.appendChild(handle);

                    handle.addEventListener("pointerdown", (event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        handle.setPointerCapture?.(event.pointerId);

                        const startX = event.clientX;
                        const startY = event.clientY;
                        const startRect = panel.getBoundingClientRect();
                        const startRight = parentWindow.innerWidth - startRect.right;
                        const startBottom = parentWindow.innerHeight - startRect.bottom;

                        const minWidth = 420;
                        const minHeight = 450;
                        const maxWidth = parentWindow.innerWidth - 32;
                        const maxHeight = parentWindow.innerHeight - 95;

                        function resize(moveEvent) {
                            const deltaX = moveEvent.clientX - startX;
                            const deltaY = moveEvent.clientY - startY;

                            let width = startRect.width;
                            let height = startRect.height;

                            if (direction === "right" || direction.includes("Right")) {
                                width = startRect.width + deltaX;
                            }
                            if (direction === "left" || direction.includes("Left")) {
                                width = startRect.width - deltaX;
                            }
                            if (direction === "bottom" || direction.includes("Bottom")) {
                                height = startRect.height + deltaY;
                            }
                            if (direction === "top" || direction.includes("Top")) {
                                height = startRect.height - deltaY;
                            }

                            width = Math.max(minWidth, Math.min(width, maxWidth));
                            height = Math.max(minHeight, Math.min(height, maxHeight));

                            panel.style.setProperty("width", `${width}px`, "important");
                            panel.style.setProperty("height", `${height}px`, "important");
                            panel.style.setProperty("right", `${startRight}px`, "important");
                            panel.style.setProperty("bottom", `${startBottom}px`, "important");
                        }

                        function stopResize() {
                            parentWindow.localStorage.setItem(storageWidth, panel.style.width);
                            parentWindow.localStorage.setItem(storageHeight, panel.style.height);
                            parentDocument.removeEventListener("pointermove", resize);
                            parentDocument.removeEventListener("pointerup", stopResize);
                            parentDocument.removeEventListener("pointercancel", stopResize);
                        }

                        parentDocument.addEventListener("pointermove", resize);
                        parentDocument.addEventListener("pointerup", stopResize);
                        parentDocument.addEventListener("pointercancel", stopResize);
                    });
                });
            }

            initializeResize();

            const observer = new MutationObserver(() => initializeResize());
            observer.observe(parentDocument.body, {childList: true, subtree: true});
        })();
        </script>
        """,
        height=0,
        width=0,
    )

def render_floating_chatbot(
    *,
    user_id: str,
    user_name: str,
    role: str,
    patient_id: str,
    patient_condition: str | None = None,
    authorized_patient_ids: list[str] | None = None,
    current_page: str = "Dashboard",
    selected_session_id: str | None = None,
    selected_exercise_id: str | None = None,
    selected_date_range: str | None = None,
    visible_metrics: dict[str, Any] | None = None,
    active_alert: dict[str, Any] | None = None,
) -> None:
    """Render a persistent bottom-right chatbot launcher and panel."""

    open_key = f"floating_chat_open::{user_id}::{patient_id}"
    if open_key not in st.session_state:
        st.session_state[open_key] = False

    st.markdown(
        """
        <style>
        .st-key-healthbridge_chat_launcher {
            position: fixed;
            right: 24px;
            bottom: 20px;
            width: 138px;
            z-index: 1000000;
        }
        .st-key-healthbridge_chat_launcher button {
            border-radius: 999px !important;
            min-height: 54px !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        }
        .st-key-healthbridge_chat_panel {
            position: fixed;
            right: 24px;
            bottom: 88px;

            /* Default size */
            width: min(650px, calc(100vw - 32px));
            height: min(820px, calc(100vh - 95px));

            /* Allowed resize limits */
            min-width: 420px;
            min-height: 450px;
            max-width: calc(100vw - 32px);
            max-height: calc(100vh - 95px);

            /* Resizing is handled by JavaScript handles on all sides. */
            overflow: auto;
            box-sizing: border-box;
            display: flex !important;
            flex-direction: column !important;

            overscroll-behavior: contain;
            z-index: 999999;
            padding: 14px 14px 18px 14px;
            border: 1px solid rgba(128, 128, 128, 0.38);
            border-radius: 16px;
            background: var(--background-color, #0e1117);
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.42);
        }
        .st-key-healthbridge_chat_panel p,
        .st-key-healthbridge_chat_panel span,
        .st-key-healthbridge_chat_panel label,
        .st-key-healthbridge_chat_panel .stMarkdown {
            font-size: 0.96rem !important;
            line-height: 1.45 !important;
        }
        .healthbridge-chat-title {
            font-weight: 800;
            font-size: 1rem;
            letter-spacing: 0.02em;
            margin: 0;
        }
        .healthbridge-chat-subtitle {
            opacity: 0.75;
            font-size: 0.82rem;
            margin-top: 2px;
        }
        .st-key-healthbridge_chat_panel [data-testid="stTextInput"] input {
            min-height: 48px !important;
        }
        .st-key-healthbridge_chat_panel [data-testid="stButton"] button {
            min-height: 48px !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        .st-key-healthbridge_chat_panel
        [data-testid="stVerticalBlockBorderWrapper"] {
            height: 100% !important;
        }
        .st-key-healthbridge_chat_panel
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            height: 100% !important;
        }
        .st-key-healthbridge_chat_panel
        div[data-testid="stHorizontalBlock"]:has(
            [data-testid="stTextInput"]
        ) {
            position: sticky !important;
            bottom: 0 !important;
            z-index: 30 !important;
            margin-top: auto !important;
            padding-top: 10px !important;
            padding-bottom: 6px !important;
            background: var(--background-color, #0e1117) !important;
            border-top: 1px solid rgba(128, 128, 128, 0.25);
        }
        @media (max-width: 600px) {
            .st-key-healthbridge_chat_launcher {
                right: 12px;
                bottom: 12px;
                width: 128px;
            }

            .st-key-healthbridge_chat_panel {
                right: 10px;
                bottom: 78px;
                width: calc(100vw - 20px);
                height: calc(100vh - 96px);
                min-width: 0;
                min-height: 0;
                resize: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state[open_key]:
        with st.container(key="healthbridge_chat_panel"):
            title_col, close_col = st.columns(
                [6, 1],
                vertical_alignment="center",
            )
            with title_col:
                st.markdown(
                    """
                    <div class="healthbridge-chat-title">HEALTHBRIDGE AI</div>
                    <div class="healthbridge-chat-subtitle">Mobility Assistant</div>
                    """,
                    unsafe_allow_html=True,
                )
            with close_col:
                if st.button(
                    "✕",
                    key=f"floating_chat_close::{user_id}::{patient_id}",
                    help="Close chat",
                ):
                    st.session_state[open_key] = False
                    st.rerun()

            st.divider()
            _render_chatbot_content(
                user_id=user_id,
                user_name=user_name,
                role=role,
                patient_id=patient_id,
                patient_condition=patient_condition,
                authorized_patient_ids=authorized_patient_ids,
                current_page=current_page,
                selected_session_id=selected_session_id,
                selected_exercise_id=selected_exercise_id,
                selected_date_range=selected_date_range,
                visible_metrics=visible_metrics,
                active_alert=active_alert,
                compact=True,
            )

        _enable_all_side_chat_resize()

    with st.container(key="healthbridge_chat_launcher"):
        button_label = (
            "Chat ⌄"
            if st.session_state[open_key]
            else "Chat 💬"
        )
        if st.button(
            button_label,
            key=f"floating_chat_toggle::{user_id}::{patient_id}",
            use_container_width=True,
            type="primary",
        ):
            st.session_state[open_key] = not st.session_state[open_key]
            st.rerun()


# Backward-compatible alias for any older imports.
def render_chatbot(**kwargs: Any) -> None:
    _render_chatbot_content(**kwargs)