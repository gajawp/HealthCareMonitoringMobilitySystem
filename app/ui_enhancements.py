"""
Chat panel enhancements for the HealthBridge mobility assistant.

Requires streamlit>=1.40 (st.audio_input and the icon= arguments).

Accessibility limits to keep in mind:
- The CSS below targets Streamlit's internal data-testid names, which can
  change between versions. Recheck it after upgrading Streamlit.
- Streamlit does not expose keyboard-focus control. Every control therefore
  has a descriptive text label, which screen readers use as its name.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Callable, TypeVar

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app.chatbot.intent import closest_sessions

__all__ = [
    "SUGGESTIONS",
    "accessibility_controls",
    "answer_with_progress",
    "closest_sessions",
    "has_native_suggestions",
    "laterality_frame",
    "play_audio",
    "render_assistant_message",
    "render_audio_with_replay",
    "render_chart",
    "render_context_header",
    "render_interpretation",
    "render_nearby_dates",
    "render_session_date_picker",
    "render_sources",
    "render_suggestion_chips",
    "render_voice_input",
    "session_question",
    "ui_text",
]

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Text
#
# One entry per chat language (same codes as chatbot_ui.LANGUAGE_OPTIONS).
# Missing keys fall back to English. A native speaker should review the
# non-English strings.
# ---------------------------------------------------------------------------

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "record_label": "Record a spoken question",
        "voice_tip": (
            "Tip: you can ask by voice. Press the microphone below, speak, "
            "then press stop. Your question is sent automatically."
        ),
        "dismiss_tip": "Got it, hide tip",
        "sources_heading": "Where this answer comes from",
        "session_word": "Session",
        "knowledge_word": "Approved guidance",
        "interpreted_as": "Interpreted as",
        "pick_date_heading": "Ask about a specific session date",
        "session_date": "Recorded session date",
        "ask_about_date": "Ask about this session",
        "nearby_heading": "Recorded sessions near that date:",
        "nearby_button": "Ask about {date}",
        "accessibility": "Accessibility",
        "text_size": "Text size",
        "size_default": "Default",
        "size_large": "Large",
        "size_xlarge": "Extra large",
        "high_contrast": "High-contrast chat",
        "speech_speed": "Speech playback speed",
        "repeat_last": "Repeat last answer aloud",
        "working": "Working on your question…",
        "done": "Answer ready",
        "failed": "Something went wrong",
        "role": "Signed in as",
        "viewing": "Viewing",
        "role_Patient": "Patient",
        "role_Caregiver": "Caregiver",
        "role_Clinician": "Clinician",
        "your_records": "Your records",
        "selected_patient": "Selected patient",
        "stage_checking": "Checking your question…",
        "stage_access": "Verifying access…",
        "stage_retrieving": "Retrieving session data…",
        "stage_guidance": "Looking up approved guidance…",
        "stage_writing": "Writing response…",
        "stage_reviewing": "Reviewing response safety…",
        "session_date_axis": "Session date",
        "left": "Left",
        "right": "Right",
        "chat_options": "Chat options",
        "send_message": "Send message",
        "language_settings": "Language and settings",
        "assistant_subtitle": "Mobility Assistant",
        "close_chat": "Close chat",
    },
    "es": {
        "record_label": "Grabar una pregunta hablada",
        "voice_tip": (
            "Consejo: puede preguntar por voz. Pulse el micrófono, hable y "
            "luego pulse detener. Su pregunta se envía automáticamente."
        ),
        "dismiss_tip": "Entendido, ocultar consejo",
        "sources_heading": "De dónde proviene esta respuesta",
        "session_word": "Sesión",
        "knowledge_word": "Guía aprobada",
        "interpreted_as": "Interpretado como",
        "pick_date_heading": "Preguntar por una fecha de sesión",
        "session_date": "Fecha de sesión registrada",
        "ask_about_date": "Preguntar por esta sesión",
        "nearby_heading": "Sesiones registradas cerca de esa fecha:",
        "nearby_button": "Preguntar por el {date}",
        "accessibility": "Accesibilidad",
        "text_size": "Tamaño del texto",
        "size_default": "Predeterminado",
        "size_large": "Grande",
        "size_xlarge": "Muy grande",
        "high_contrast": "Chat de alto contraste",
        "speech_speed": "Velocidad de reproducción de voz",
        "repeat_last": "Repetir la última respuesta en voz alta",
        "working": "Procesando su pregunta…",
        "done": "Respuesta lista",
        "failed": "Algo salió mal",
        "role": "Sesión iniciada como",
        "viewing": "Viendo",
        "role_Patient": "Paciente",
        "role_Caregiver": "Cuidador",
        "role_Clinician": "Clínico",
        "your_records": "Sus registros",
        "selected_patient": "Paciente seleccionado",
        "stage_checking": "Revisando su pregunta…",
        "stage_access": "Verificando el acceso…",
        "stage_retrieving": "Obteniendo datos de la sesión…",
        "stage_guidance": "Buscando la guía aprobada…",
        "stage_writing": "Redactando la respuesta…",
        "stage_reviewing": "Revisando la seguridad de la respuesta…",
        "session_date_axis": "Fecha de sesión",
        "left": "Izquierda",
        "right": "Derecha",
        "chat_options": "Opciones del chat",
        "send_message": "Enviar mensaje",
        "language_settings": "Idioma y configuración",
        "assistant_subtitle": "Asistente de movilidad",
        "close_chat": "Cerrar chat",
    },
    "fr": {
        "record_label": "Enregistrer une question orale",
        "voice_tip": (
            "Astuce : vous pouvez poser votre question à voix haute. Appuyez "
            "sur le micro ci-dessous, parlez, puis appuyez sur stop. Votre "
            "question est envoyée automatiquement."
        ),
        "dismiss_tip": "Compris, masquer l’astuce",
        "sources_heading": "D’où vient cette réponse",
        "session_word": "Séance",
        "knowledge_word": "Conseils approuvés",
        "interpreted_as": "Interprété comme",
        "pick_date_heading": "Demander une date de séance précise",
        "session_date": "Date de séance enregistrée",
        "ask_about_date": "Demander sur cette séance",
        "nearby_heading": "Séances enregistrées proches de cette date :",
        "nearby_button": "Demander le {date}",
        "accessibility": "Accessibilité",
        "text_size": "Taille du texte",
        "size_default": "Par défaut",
        "size_large": "Grand",
        "size_xlarge": "Très grand",
        "high_contrast": "Chat à contraste élevé",
        "speech_speed": "Vitesse de lecture vocale",
        "repeat_last": "Réécouter la dernière réponse",
        "working": "Traitement de votre question…",
        "done": "Réponse prête",
        "failed": "Une erreur s’est produite",
        "role": "Connecté en tant que",
        "viewing": "Dossier consulté",
        "role_Patient": "Patient",
        "role_Caregiver": "Aidant",
        "role_Clinician": "Clinicien",
        "your_records": "Votre dossier",
        "selected_patient": "Patient sélectionné",
        "stage_checking": "Vérification de votre question…",
        "stage_access": "Vérification des accès…",
        "stage_retrieving": "Récupération des données de séance…",
        "stage_guidance": "Recherche des conseils approuvés…",
        "stage_writing": "Rédaction de la réponse…",
        "stage_reviewing": "Vérification de la sécurité de la réponse…",
        "session_date_axis": "Date de séance",
        "left": "Gauche",
        "right": "Droite",
        "chat_options": "Options du chat",
        "send_message": "Envoyer le message",
        "language_settings": "Langue et paramètres",
        "assistant_subtitle": "Assistant de mobilité",
        "close_chat": "Fermer le chat",
    },
    "de": {
        "record_label": "Gesprochene Frage aufnehmen",
        "voice_tip": (
            "Tipp: Sie können per Sprache fragen. Tippen Sie unten auf das "
            "Mikrofon, sprechen Sie und tippen Sie dann auf Stopp. Ihre Frage "
            "wird automatisch gesendet."
        ),
        "dismiss_tip": "Verstanden, Tipp ausblenden",
        "sources_heading": "Woher diese Antwort stammt",
        "session_word": "Sitzung",
        "knowledge_word": "Geprüfte Anleitung",
        "interpreted_as": "Verstanden als",
        "pick_date_heading": "Nach einem bestimmten Sitzungsdatum fragen",
        "session_date": "Aufgezeichnetes Sitzungsdatum",
        "ask_about_date": "Nach dieser Sitzung fragen",
        "nearby_heading": "Aufgezeichnete Sitzungen nahe diesem Datum:",
        "nearby_button": "Nach {date} fragen",
        "accessibility": "Barrierefreiheit",
        "text_size": "Textgröße",
        "size_default": "Standard",
        "size_large": "Groß",
        "size_xlarge": "Sehr groß",
        "high_contrast": "Chat mit hohem Kontrast",
        "speech_speed": "Sprachwiedergabe-Geschwindigkeit",
        "repeat_last": "Letzte Antwort erneut vorlesen",
        "working": "Ihre Frage wird bearbeitet…",
        "done": "Antwort bereit",
        "failed": "Etwas ist schiefgelaufen",
        "role": "Angemeldet als",
        "viewing": "Ansicht",
        "role_Patient": "Patient/in",
        "role_Caregiver": "Pflegeperson",
        "role_Clinician": "Ärztin/Arzt",
        "your_records": "Ihre Daten",
        "selected_patient": "Ausgewählte Patientin / ausgewählter Patient",
        "stage_checking": "Ihre Frage wird geprüft…",
        "stage_access": "Zugriff wird überprüft…",
        "stage_retrieving": "Sitzungsdaten werden abgerufen…",
        "stage_guidance": "Geprüfte Anleitungen werden gesucht…",
        "stage_writing": "Antwort wird verfasst…",
        "stage_reviewing": "Antwort wird auf Sicherheit geprüft…",
        "session_date_axis": "Sitzungsdatum",
        "left": "Links",
        "right": "Rechts",
        "chat_options": "Chat-Optionen",
        "send_message": "Nachricht senden",
        "language_settings": "Sprache und Einstellungen",
        "assistant_subtitle": "Mobilitätsassistent",
        "close_chat": "Chat schließen",
    },
    "hi": {
        "record_label": "बोलकर प्रश्न रिकॉर्ड करें",
        "voice_tip": (
            "सुझाव: आप बोलकर पूछ सकते हैं। नीचे माइक्रोफ़ोन दबाएँ, बोलें, फिर "
            "रोकें दबाएँ। आपका प्रश्न अपने आप भेज दिया जाएगा।"
        ),
        "dismiss_tip": "समझ गया, सुझाव छिपाएँ",
        "sources_heading": "यह उत्तर कहाँ से आया है",
        "session_word": "सत्र",
        "knowledge_word": "स्वीकृत मार्गदर्शन",
        "interpreted_as": "इस रूप में समझा गया",
        "pick_date_heading": "किसी विशेष सत्र तिथि के बारे में पूछें",
        "session_date": "रिकॉर्ड किए गए सत्र की तिथि",
        "ask_about_date": "इस सत्र के बारे में पूछें",
        "nearby_heading": "उस तिथि के आसपास रिकॉर्ड किए गए सत्र:",
        "nearby_button": "{date} के बारे में पूछें",
        "accessibility": "सुलभता",
        "text_size": "टेक्स्ट का आकार",
        "size_default": "सामान्य",
        "size_large": "बड़ा",
        "size_xlarge": "बहुत बड़ा",
        "high_contrast": "उच्च कंट्रास्ट चैट",
        "speech_speed": "आवाज़ चलाने की गति",
        "repeat_last": "पिछला उत्तर फिर से सुनें",
        "working": "आपके प्रश्न पर काम हो रहा है…",
        "done": "उत्तर तैयार है",
        "failed": "कुछ गलत हो गया",
        "role": "इस रूप में साइन इन",
        "viewing": "देख रहे हैं",
        "role_Patient": "रोगी",
        "role_Caregiver": "देखभालकर्ता",
        "role_Clinician": "चिकित्सक",
        "your_records": "आपके रिकॉर्ड",
        "selected_patient": "चयनित रोगी",
        "stage_checking": "आपके प्रश्न की जाँच हो रही है…",
        "stage_access": "पहुँच की पुष्टि हो रही है…",
        "stage_retrieving": "सत्र का डेटा प्राप्त किया जा रहा है…",
        "stage_guidance": "स्वीकृत मार्गदर्शन खोजा जा रहा है…",
        "stage_writing": "उत्तर लिखा जा रहा है…",
        "stage_reviewing": "उत्तर की सुरक्षा जाँची जा रही है…",
        "session_date_axis": "सत्र तिथि",
        "left": "बायाँ",
        "right": "दायाँ",
        "chat_options": "चैट विकल्प",
        "send_message": "संदेश भेजें",
        "language_settings": "भाषा और सेटिंग्स",
        "assistant_subtitle": "मोबिलिटी सहायक",
        "close_chat": "चैट बंद करें",
    },
    "te": {
        "record_label": "మాట్లాడి ప్రశ్నను రికార్డ్ చేయండి",
        "voice_tip": (
            "చిట్కా: మీరు మాటల ద్వారా అడగవచ్చు. కింద ఉన్న మైక్రోఫోన్‌ను నొక్కి, "
            "మాట్లాడి, ఆపై ఆపు నొక్కండి. మీ ప్రశ్న స్వయంచాలకంగా పంపబడుతుంది."
        ),
        "dismiss_tip": "అర్థమైంది, చిట్కాను దాచు",
        "sources_heading": "ఈ సమాధానం ఎక్కడి నుంచి వచ్చింది",
        "session_word": "సెషన్",
        "knowledge_word": "ఆమోదిత మార్గదర్శకం",
        "interpreted_as": "ఇలా అర్థం చేసుకున్నాం",
        "pick_date_heading": "నిర్దిష్ట సెషన్ తేదీ గురించి అడగండి",
        "session_date": "రికార్డ్ చేసిన సెషన్ తేదీ",
        "ask_about_date": "ఈ సెషన్ గురించి అడగండి",
        "nearby_heading": "ఆ తేదీకి దగ్గరగా రికార్డ్ చేసిన సెషన్లు:",
        "nearby_button": "{date} గురించి అడగండి",
        "accessibility": "సులభ వినియోగం",
        "text_size": "అక్షరాల పరిమాణం",
        "size_default": "సాధారణ",
        "size_large": "పెద్దది",
        "size_xlarge": "చాలా పెద్దది",
        "high_contrast": "అధిక కాంట్రాస్ట్ చాట్",
        "speech_speed": "వాయిస్ ప్లేబ్యాక్ వేగం",
        "repeat_last": "చివరి సమాధానాన్ని మళ్లీ వినండి",
        "working": "మీ ప్రశ్నపై పని జరుగుతోంది…",
        "done": "సమాధానం సిద్ధంగా ఉంది",
        "failed": "ఏదో పొరపాటు జరిగింది",
        "role": "ఇలా సైన్ ఇన్ అయ్యారు",
        "viewing": "చూస్తున్నది",
        "role_Patient": "రోగి",
        "role_Caregiver": "సంరక్షకులు",
        "role_Clinician": "వైద్యులు",
        "your_records": "మీ రికార్డులు",
        "selected_patient": "ఎంచుకున్న రోగి",
        "stage_checking": "మీ ప్రశ్నను తనిఖీ చేస్తున్నాం…",
        "stage_access": "యాక్సెస్‌ను నిర్ధారిస్తున్నాం…",
        "stage_retrieving": "సెషన్ డేటాను తీసుకుంటున్నాం…",
        "stage_guidance": "ఆమోదిత మార్గదర్శకాన్ని వెతుకుతున్నాం…",
        "stage_writing": "సమాధానం రాస్తున్నాం…",
        "stage_reviewing": "సమాధానం భద్రతను సమీక్షిస్తున్నాం…",
        "session_date_axis": "సెషన్ తేదీ",
        "left": "ఎడమ",
        "right": "కుడి",
        "chat_options": "చాట్ ఎంపికలు",
        "send_message": "సందేశం పంపండి",
        "language_settings": "భాష మరియు సెట్టింగ్‌లు",
        "assistant_subtitle": "మొబిలిటీ సహాయకుడు",
        "close_chat": "చాట్ మూసివేయండి",
    },
    "ta": {
        "record_label": "பேசிக் கேள்வியைப் பதிவு செய்யவும்",
        "voice_tip": (
            "குறிப்பு: நீங்கள் குரல் மூலம் கேட்கலாம். கீழே உள்ள மைக்ரோஃபோனை "
            "அழுத்தி, பேசி, பின்னர் நிறுத்து என்பதை அழுத்தவும். உங்கள் கேள்வி "
            "தானாக அனுப்பப்படும்."
        ),
        "dismiss_tip": "புரிந்தது, குறிப்பை மறை",
        "sources_heading": "இந்தப் பதில் எங்கிருந்து வருகிறது",
        "session_word": "அமர்வு",
        "knowledge_word": "அங்கீகரிக்கப்பட்ட வழிகாட்டல்",
        "interpreted_as": "இவ்வாறு புரிந்துகொள்ளப்பட்டது",
        "pick_date_heading": "குறிப்பிட்ட அமர்வு தேதி பற்றிக் கேளுங்கள்",
        "session_date": "பதிவுசெய்யப்பட்ட அமர்வு தேதி",
        "ask_about_date": "இந்த அமர்வு பற்றிக் கேளுங்கள்",
        "nearby_heading": "அந்தத் தேதிக்கு அருகில் பதிவுசெய்யப்பட்ட அமர்வுகள்:",
        "nearby_button": "{date} பற்றிக் கேளுங்கள்",
        "accessibility": "அணுகல்தன்மை",
        "text_size": "எழுத்து அளவு",
        "size_default": "இயல்புநிலை",
        "size_large": "பெரியது",
        "size_xlarge": "மிகப் பெரியது",
        "high_contrast": "அதிக மாறுபாடு உரையாடல்",
        "speech_speed": "குரல் இயக்க வேகம்",
        "repeat_last": "கடைசி பதிலை மீண்டும் கேளுங்கள்",
        "working": "உங்கள் கேள்வி செயலாக்கப்படுகிறது…",
        "done": "பதில் தயார்",
        "failed": "ஏதோ தவறு நடந்தது",
        "role": "இவ்வாறு உள்நுழைந்துள்ளீர்கள்",
        "viewing": "பார்ப்பது",
        "role_Patient": "நோயாளி",
        "role_Caregiver": "பராமரிப்பாளர்",
        "role_Clinician": "மருத்துவர்",
        "your_records": "உங்கள் பதிவுகள்",
        "selected_patient": "தேர்ந்தெடுத்த நோயாளி",
        "stage_checking": "உங்கள் கேள்வி சரிபார்க்கப்படுகிறது…",
        "stage_access": "அணுகல் உறுதிசெய்யப்படுகிறது…",
        "stage_retrieving": "அமர்வுத் தரவு பெறப்படுகிறது…",
        "stage_guidance": "அங்கீகரிக்கப்பட்ட வழிகாட்டல் தேடப்படுகிறது…",
        "stage_writing": "பதில் எழுதப்படுகிறது…",
        "stage_reviewing": "பதிலின் பாதுகாப்பு சரிபார்க்கப்படுகிறது…",
        "session_date_axis": "அமர்வு தேதி",
        "left": "இடது",
        "right": "வலது",
        "chat_options": "உரையாடல் விருப்பங்கள்",
        "send_message": "செய்தியை அனுப்பு",
        "language_settings": "மொழி மற்றும் அமைப்புகள்",
        "assistant_subtitle": "இயக்கத்திறன் உதவியாளர்",
        "close_chat": "உரையாடலை மூடு",
    },
    "zh": {
        "record_label": "录制语音问题",
        "voice_tip": "提示：您可以用语音提问。按下方的麦克风，说话，然后按停止。您的问题会自动发送。",
        "dismiss_tip": "知道了，隐藏提示",
        "sources_heading": "此回答的来源",
        "session_word": "训练",
        "knowledge_word": "已批准的指导",
        "interpreted_as": "理解为",
        "pick_date_heading": "询问特定训练日期",
        "session_date": "已记录的训练日期",
        "ask_about_date": "询问此次训练",
        "nearby_heading": "该日期附近已记录的训练：",
        "nearby_button": "询问 {date}",
        "accessibility": "无障碍",
        "text_size": "文字大小",
        "size_default": "默认",
        "size_large": "大",
        "size_xlarge": "特大",
        "high_contrast": "高对比度聊天",
        "speech_speed": "语音播放速度",
        "repeat_last": "再次朗读上一条回答",
        "working": "正在处理您的问题…",
        "done": "回答已就绪",
        "failed": "出现错误",
        "role": "登录身份",
        "viewing": "正在查看",
        "role_Patient": "患者",
        "role_Caregiver": "护理人员",
        "role_Clinician": "临床医生",
        "your_records": "您的记录",
        "selected_patient": "所选患者",
        "stage_checking": "正在检查您的问题…",
        "stage_access": "正在验证访问权限…",
        "stage_retrieving": "正在获取训练数据…",
        "stage_guidance": "正在查找已批准的指导…",
        "stage_writing": "正在撰写回答…",
        "stage_reviewing": "正在审核回答安全性…",
        "session_date_axis": "训练日期",
        "left": "左",
        "right": "右",
        "chat_options": "聊天选项",
        "send_message": "发送消息",
        "language_settings": "语言和设置",
        "assistant_subtitle": "活动能力助手",
        "close_chat": "关闭聊天",
    },
    "ar": {
        "record_label": "سجّل سؤالًا منطوقًا",
        "voice_tip": (
            "نصيحة: يمكنك السؤال بالصوت. اضغط على الميكروفون أدناه، وتحدّث، "
            "ثم اضغط إيقاف. سيُرسَل سؤالك تلقائيًا."
        ),
        "dismiss_tip": "فهمت، إخفاء النصيحة",
        "sources_heading": "مصدر هذه الإجابة",
        "session_word": "الجلسة",
        "knowledge_word": "إرشادات معتمدة",
        "interpreted_as": "فُهم على أنه",
        "pick_date_heading": "اسأل عن تاريخ جلسة محدد",
        "session_date": "تاريخ الجلسة المسجلة",
        "ask_about_date": "اسأل عن هذه الجلسة",
        "nearby_heading": "جلسات مسجلة قريبة من ذلك التاريخ:",
        "nearby_button": "اسأل عن {date}",
        "accessibility": "إمكانية الوصول",
        "text_size": "حجم النص",
        "size_default": "افتراضي",
        "size_large": "كبير",
        "size_xlarge": "كبير جدًا",
        "high_contrast": "دردشة عالية التباين",
        "speech_speed": "سرعة تشغيل الصوت",
        "repeat_last": "أعد قراءة الإجابة الأخيرة",
        "working": "جارٍ العمل على سؤالك…",
        "done": "الإجابة جاهزة",
        "failed": "حدث خطأ ما",
        "role": "تم تسجيل الدخول بصفة",
        "viewing": "السجل المعروض",
        "role_Patient": "مريض",
        "role_Caregiver": "مقدّم رعاية",
        "role_Clinician": "طبيب",
        "your_records": "سجلاتك",
        "selected_patient": "المريض المحدد",
        "stage_checking": "جارٍ فحص سؤالك…",
        "stage_access": "جارٍ التحقق من الصلاحية…",
        "stage_retrieving": "جارٍ استرجاع بيانات الجلسة…",
        "stage_guidance": "جارٍ البحث عن الإرشادات المعتمدة…",
        "stage_writing": "جارٍ كتابة الإجابة…",
        "stage_reviewing": "جارٍ مراجعة سلامة الإجابة…",
        "session_date_axis": "تاريخ الجلسة",
        "left": "يسار",
        "right": "يمين",
        "chat_options": "خيارات الدردشة",
        "send_message": "إرسال الرسالة",
        "language_settings": "اللغة والإعدادات",
        "assistant_subtitle": "مساعد الحركة",
        "close_chat": "إغلاق الدردشة",
    },
}


def _t(language: str, key: str) -> str:
    return TEXT.get(language, TEXT["en"]).get(key, TEXT["en"][key])


def ui_text(language: str, key: str) -> str:
    """Localized fixed UI text for the chat panel."""

    return _t(language, key)


# Orchestrator progress messages (English) mapped to translated stages.
STAGE_KEYS: dict[str, str] = {
    "Checking your question…": "stage_checking",
    "Verifying access…": "stage_access",
    "Retrieving session data…": "stage_retrieving",
    "Looking up approved guidance…": "stage_guidance",
    "Writing response…": "stage_writing",
    "Reviewing response safety…": "stage_reviewing",
}


METRIC_NAMES: dict[str, str] = {
    "duration_s": "repetition duration",
    "lift_duration_s": "lift duration",
    "lower_duration_s": "lowering duration",
    "hold_duration_s": "hold duration",
    "shin_lift_angle_deg": "shin lift angle",
    "peak_angular_velocity": "peak angular velocity",
    "jerk_score": "jerk score",
    "tof_symmetry": "toe-off symmetry",
}


def _pretty_date(value: str | None, language: str = "en") -> str:
    """"Mar 14, 2026" in English; ISO "2026-03-14" in other languages."""

    try:
        parsed = date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return str(value or "")
    if language != "en":
        return parsed.isoformat()
    return f"{parsed:%b} {parsed.day}, {parsed.year}"


# ---------------------------------------------------------------------------
# 1. Voice input
# ---------------------------------------------------------------------------


def render_voice_input(*, key: str, language: str = "en") -> bytes | None:
    """
    Render a labelled recorder and return new audio bytes once.

    st.audio_input shows its own recording state and timer. Each recording
    is hashed so a rerun never submits the same audio twice.
    """

    tip_key = f"{key}::voice_tip_dismissed"

    if not st.session_state.get(tip_key):
        st.info(_t(language, "voice_tip"), icon="🎙️")
        if st.button(
            _t(language, "dismiss_tip"),
            key=f"{key}::dismiss_voice_tip",
            type="tertiary",
        ):
            st.session_state[tip_key] = True
            st.rerun()

    recording = st.audio_input(
        _t(language, "record_label"),
        key=f"{key}::recorder",
    )

    if recording is None:
        return None

    audio_bytes = recording.getvalue()
    if not audio_bytes:
        return None

    digest = hashlib.sha256(audio_bytes).hexdigest()
    digest_key = f"{key}::last_recording_digest"

    if st.session_state.get(digest_key) == digest:
        return None

    st.session_state[digest_key] = digest
    return audio_bytes


# ---------------------------------------------------------------------------
# 2. Source citations
# ---------------------------------------------------------------------------


def render_sources(
    session_sources: list[dict[str, Any]] | None,
    knowledge_sources: list[str] | None = None,
    *,
    language: str = "en",
    translate: Callable[[str], str] | None = None,
) -> None:
    """
    Show the sessions, metrics and approved documents behind an answer.

    Sessions are listed by internal session token only, never by patient
    identifier, consistent with the de-identification design.
    """

    session_sources = session_sources or []
    knowledge_sources = knowledge_sources or []

    if not session_sources and not knowledge_sources:
        return

    with st.expander(_t(language, "sources_heading"), icon="🔎"):
        for entry in session_sources:
            metrics = ", ".join(
                METRIC_NAMES.get(metric, metric)
                for metric in entry.get("metrics", [])
            )
            if metrics and translate and language != "en":
                metrics = translate(metrics)
            date_text = _pretty_date(entry.get("session_date"), language)
            st.markdown(
                f"- **{_t(language, 'session_word')} {date_text}** · "
                f"`{entry.get('session_id', '')}`"
                + (f" — {metrics}" if metrics else "")
            )

        for source in knowledge_sources:
            st.markdown(f"- {_t(language, 'knowledge_word')}: `{source}`")


# ---------------------------------------------------------------------------
# 3. Date handling
# ---------------------------------------------------------------------------


def session_question(session_date: str) -> str:
    """Return a cleanly worded question for one recorded session date."""

    return f"Summarize the mobility session on {session_date}."


def render_interpretation(
    interpretation: str | None,
    *,
    language: str = "en",
) -> None:
    """Show how dates or "latest" were resolved so misreads are obvious."""

    if interpretation:
        st.caption(f"🧭 {_t(language, 'interpreted_as')}: **{interpretation}**")


def render_session_date_picker(
    session_dates: list[str],
    *,
    key: str,
    language: str = "en",
) -> str | None:
    """
    Let the user pick a recorded session date and return a ready question.

    Picking from real dates avoids free-text date parsing entirely.
    """

    if not session_dates:
        return None

    with st.expander(_t(language, "pick_date_heading"), icon="📅"):
        chosen = st.selectbox(
            _t(language, "session_date"),
            options=sorted(session_dates, reverse=True),
            format_func=lambda value: _pretty_date(value, language),
            key=f"{key}::session_date",
        )

        if st.button(
            _t(language, "ask_about_date"),
            key=f"{key}::ask_session_date",
            icon="📨",
        ):
            return session_question(chosen)

    return None


def render_nearby_dates(
    nearby_dates: list[str] | None,
    *,
    key: str,
    language: str = "en",
) -> str | None:
    """Render one-tap buttons for sessions close to a date with no data."""

    if not nearby_dates:
        return None

    st.caption(_t(language, "nearby_heading"))
    columns = st.columns(len(nearby_dates))
    selected: str | None = None

    for column, session_date in zip(columns, nearby_dates):
        with column:
            if st.button(
                _t(language, "nearby_button").format(
                    date=_pretty_date(session_date, language)
                ),
                key=f"{key}::nearby::{session_date}",
                icon="📅",
                use_container_width=True,
            ):
                selected = session_question(session_date)

    return selected


# ---------------------------------------------------------------------------
# 4. Suggestion chips
#
# Shape: SUGGESTIONS[language][role] = [(icon, label, full question), ...]
# The "_exercise" entry is a template used when an exercise is selected.
# ---------------------------------------------------------------------------

SUGGESTIONS: dict[str, dict[str, list[tuple[str, str, str]]]] = {
    "en": {
        "Patient": [
            ("🏋️", "My suggested exercises", "Show my suggested exercises."),
            ("📋", "My latest session", "Summarize my latest mobility session."),
            ("⚠️", "Explain my alerts", "Why were my repetitions flagged?"),
            ("📈", "Am I improving?", "Am I improving over time?"),
        ],
        "Caregiver": [
            ("🏋️", "Suggested exercises", "Show the suggested exercises for this patient."),
            ("📋", "Latest session", "Summarize the latest mobility session."),
            ("⚠️", "Explain alerts", "Why were repetitions flagged?"),
            ("📈", "Progress", "Is the patient improving over time?"),
        ],
        "Clinician": [
            ("🏋️", "Suggested exercises", "Show the suggested exercises for this patient."),
            ("📋", "Session summary", "Summarize the latest recorded mobility session."),
            ("⚠️", "Flagged repetitions", "Explain the flagged repetitions."),
            ("📐", "Metric explanation", "What does jerk score mean?"),
        ],
        "_exercise": [
            ("🦵", "Guidance: {exercise}", "How should I perform {exercise}?"),
        ],
    },
    "es": {
        "Patient": [
            ("🏋️", "Mis ejercicios sugeridos", "Muestra mis ejercicios sugeridos."),
            ("📋", "Mi última sesión", "Resume mi última sesión de movilidad."),
            ("⚠️", "Explicar mis alertas", "¿Por qué se marcaron mis repeticiones?"),
            ("📈", "¿Estoy mejorando?", "¿Estoy mejorando con el tiempo?"),
        ],
        "Caregiver": [
            ("🏋️", "Ejercicios sugeridos", "Muestra los ejercicios sugeridos para este paciente."),
            ("📋", "Última sesión", "Resume la última sesión de movilidad."),
            ("⚠️", "Explicar alertas", "¿Por qué se marcaron las repeticiones?"),
            ("📈", "Progreso", "¿Está mejorando el paciente con el tiempo?"),
        ],
        "Clinician": [
            ("🏋️", "Ejercicios sugeridos", "Muestra los ejercicios sugeridos para este paciente."),
            ("📋", "Resumen de sesión", "Resume la última sesión de movilidad registrada."),
            ("⚠️", "Repeticiones marcadas", "Explica las repeticiones marcadas."),
            ("📐", "Explicación de métricas", "¿Qué significa la puntuación de sacudida (jerk score)?"),
        ],
        "_exercise": [
            ("🦵", "Guía: {exercise}", "¿Cómo debo realizar {exercise}?"),
        ],
    },
}


def has_native_suggestions(language: str) -> bool:
    return language in SUGGESTIONS


def render_suggestion_chips(
    role: str,
    *,
    key: str,
    language: str = "en",
    exercise: str | None = None,
    limit: int | None = None,
    columns: int = 2,
    translate_label: Callable[[str], str] | None = None,
) -> str | None:
    """
    Render suggestion chips and return the clicked chip's full question.

    Languages missing from SUGGESTIONS use the English chips; pass
    translate_label to localize their labels. The returned question is in
    the chip's own language, so callers should send it through the same
    pipeline as a typed question.
    """

    chip_set = SUGGESTIONS.get(language, SUGGESTIONS["en"])
    chips = list(chip_set.get(role, chip_set["Patient"]))

    if exercise and role in {"Caregiver", "Clinician"}:
        chips.extend(
            (
                icon,
                label.format(exercise=exercise),
                question.format(exercise=exercise),
            )
            for icon, label, question in chip_set["_exercise"]
        )

    if limit is not None:
        chips = chips[:limit]

    grid = st.columns(max(1, columns))
    selected: str | None = None

    for index, (icon, label, question) in enumerate(chips):
        display_label = (
            translate_label(label)
            if translate_label and not has_native_suggestions(language)
            else label
        )
        with grid[index % len(grid)]:
            if st.button(
                display_label,
                icon=icon,
                key=f"{key}::chip::{index}",
                help=question,
                use_container_width=True,
            ):
                selected = question

    return selected


# ---------------------------------------------------------------------------
# 5. Inline charts
# ---------------------------------------------------------------------------


def laterality_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Turn per-session left/right rows into a chart-ready DataFrame.

    Expected rows: {"session": "YYYY-MM-DD", "Left": float, "Right": float}.
    Rename these keys here if your session record fields differ.
    """

    frame = pd.DataFrame(rows)
    if frame.empty or "session" not in frame.columns:
        return pd.DataFrame()

    frame["session"] = pd.to_datetime(frame["session"], errors="coerce")
    frame = frame.dropna(subset=["session"]).sort_values("session")

    legs = [column for column in ("Left", "Right") if column in frame.columns]
    return frame.set_index("session")[legs]


def render_chart(
    chart_data: dict[str, Any] | None,
    *,
    compact: bool = False,
    language: str = "en",
    translate: Callable[[str], str] | None = None,
) -> None:
    """Render a left/right chart: bars for one or two sessions, lines for trends."""

    if not chart_data or not chart_data.get("rows"):
        return

    frame = laterality_frame(chart_data["rows"])
    if frame.empty:
        return

    title = chart_data.get("title", "")
    if title and translate and language != "en":
        title = translate(title)

    frame = frame.rename(
        columns={"Left": _t(language, "left"), "Right": _t(language, "right")}
    )
    x_label = _t(language, "session_date_axis")

    st.caption(f"📊 {title}")
    height = 220 if compact else 300

    if chart_data.get("chart_kind") == "line":
        st.line_chart(frame, height=height, x_label=x_label)
    else:
        frame.index = frame.index.strftime("%Y-%m-%d")
        st.bar_chart(frame, height=height, stack=False, x_label=x_label)


# ---------------------------------------------------------------------------
# 6. Accessibility
# ---------------------------------------------------------------------------

TEXT_SIZES: dict[str, float] = {
    "Default": 1.0,
    "Large": 1.15,
    "Extra large": 1.3,
}

TEXT_SIZE_KEYS: dict[str, str] = {
    "Default": "size_default",
    "Large": "size_large",
    "Extra large": "size_xlarge",
}

SPEECH_SPEEDS: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5)

_CHAT_SCOPE = (
    '[data-testid="stChatMessage"]',
    ".st-key-healthbridge_chat_panel",
)


def _accessibility_css(
    scale: float,
    high_contrast: bool,
    page_base_px: float | None = None,
) -> str:
    text_selectors = ", ".join(
        f"{scope} {element}"
        for scope in _CHAT_SCOPE
        for element in ("p", "li", "span", "label", ".stMarkdown")
    )

    css = ""
    chat_scale = scale

    # Page-wide: override the dashboard's fixed base font size. Chat text is
    # sized in rem, so it scales with the page and needs no extra factor.
    if page_base_px is not None:
        css += (
            'html, body, [class*="st-"] {'
            f" font-size: {page_base_px * scale:.1f}px !important; }}"
        )
        chat_scale = 1.0

    css += (
        f"{text_selectors} {{"
        f" font-size: {0.96 * chat_scale:.2f}rem !important;"
        f" line-height: 1.5 !important; }}"
    )

    if high_contrast:
        css += """
        [data-testid="stChatMessage"] {
            background: #000000 !important;
            color: #ffffff !important;
            border: 2px solid #ffffff !important;
        }
        [data-testid="stChatMessage"] * { color: #ffffff !important; }
        .st-key-healthbridge_chat_panel {
            background: #000000 !important;
            border: 2px solid #ffff00 !important;
        }
        .st-key-healthbridge_chat_panel [data-testid="stButton"] button,
        .st-key-healthbridge_chat_panel [data-testid="stExpander"] {
            border: 2px solid #ffff00 !important;
        }
        .st-key-healthbridge_chat_panel button:focus-visible,
        .st-key-healthbridge_chat_panel input:focus-visible {
            outline: 3px solid #ffff00 !important;
            outline-offset: 2px !important;
        }
        """

    return f"<style>{css}</style>"


def _apply_speech_rate(rate: float) -> None:
    """
    Set the playback rate of every audio element on the page.

    Works with any text-to-speech output because it only changes the
    browser's playbackRate, including players added after this runs.
    """

    components.html(
        f"""
        <script>
        (() => {{
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            const rate = {json.dumps(rate)};
            const apply = (audio) => {{
                audio.defaultPlaybackRate = rate;
                audio.playbackRate = rate;
            }};

            parentWindow.__healthbridgeRateObserver?.disconnect();
            if (parentWindow.__healthbridgeRateListener) {{
                doc.removeEventListener(
                    "play", parentWindow.__healthbridgeRateListener, true
                );
            }}

            const listener = (event) => {{
                if (event.target instanceof parentWindow.HTMLAudioElement) {{
                    apply(event.target);
                }}
            }};
            doc.addEventListener("play", listener, true);
            parentWindow.__healthbridgeRateListener = listener;

            doc.querySelectorAll("audio").forEach(apply);
            const observer = new MutationObserver(
                () => doc.querySelectorAll("audio").forEach(apply)
            );
            observer.observe(doc.body, {{childList: true, subtree: true}});
            parentWindow.__healthbridgeRateObserver = observer;
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def accessibility_controls(
    *,
    key: str,
    language: str = "en",
    page_base_px: float | None = None,
) -> dict[str, Any]:
    """
    Render text size, high contrast and speech speed in the sidebar.

    With page_base_px (the dashboard's base font size), text size applies
    to the whole page; otherwise only to the chat.

    Returns {"text_scale": float, "high_contrast": bool, "speech_rate": float}.
    """

    with st.sidebar:
        st.markdown(f"### ♿ {_t(language, 'accessibility')}")

        size_name = st.select_slider(
            _t(language, "text_size"),
            options=list(TEXT_SIZES),
            value="Default",
            format_func=lambda name: _t(language, TEXT_SIZE_KEYS[name]),
            key=f"{key}::text_size",
        )
        high_contrast = st.toggle(
            _t(language, "high_contrast"),
            value=False,
            key=f"{key}::high_contrast",
        )
        speech_rate = st.select_slider(
            _t(language, "speech_speed"),
            options=list(SPEECH_SPEEDS),
            value=1.0,
            format_func=lambda value: f"{value:g}×",
            key=f"{key}::speech_rate",
        )

    scale = TEXT_SIZES[size_name]
    st.markdown(
        _accessibility_css(scale, high_contrast, page_base_px),
        unsafe_allow_html=True,
    )
    _apply_speech_rate(speech_rate)

    return {
        "text_scale": scale,
        "high_contrast": high_contrast,
        "speech_rate": speech_rate,
    }


def play_audio(audio_bytes: bytes | None, *, autoplay: bool = False) -> None:
    """Play TTS audio; the speed set in accessibility_controls applies."""

    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay)


def render_audio_with_replay(
    *,
    state_key: str,
    language: str = "en",
) -> None:
    """
    Render a "Repeat last answer" button for the audio in st.session_state.

    Store the latest answer's audio bytes under state_key. Scope the key to
    the signed-in user and selected patient so audio is never replayed
    across patients.
    """

    last_audio = st.session_state.get(state_key)
    if not last_audio:
        return

    if st.button(
        _t(language, "repeat_last"),
        key=f"{state_key}::repeat",
        icon="🔁",
    ):
        play_audio(last_audio, autoplay=True)


# ---------------------------------------------------------------------------
# 7. Progress feedback
# ---------------------------------------------------------------------------


def answer_with_progress(
    run: Callable[[Callable[[str], None]], T],
    *,
    language: str = "en",
) -> T:
    """
    Run the pipeline while showing its live stage messages.

    ``run`` receives a report(message) callback to pass to the orchestrator.
    The full answer is still checked by the guardrails before it is shown;
    this shows progress, not partial text.
    """

    with st.status(_t(language, "working"), expanded=False) as status:
        try:
            result = run(
                lambda message: status.update(
                    label=_t(language, STAGE_KEYS[message])
                    if message in STAGE_KEYS
                    else message
                )
            )
        except Exception:
            status.update(label=_t(language, "failed"), state="error")
            raise
        status.update(label=_t(language, "done"), state="complete")

    return result


# ---------------------------------------------------------------------------
# 8. Context header
# ---------------------------------------------------------------------------


def render_context_header(
    role: str,
    patient_label: str | None,
    *,
    language: str = "en",
) -> None:
    """
    Show who is signed in and whose record the chat is about.

    patient_label is a display name; None shows "Your records" for a
    patient and "Selected patient" otherwise, in the chat language.
    """

    role_text = TEXT.get(language, TEXT["en"]).get(f"role_{role}", role)
    if patient_label is None:
        patient_label = _t(
            language,
            "your_records" if role == "Patient" else "selected_patient",
        )

    st.caption(
        f"👤 {_t(language, 'role')} **{role_text}** · "
        f"🩺 {_t(language, 'viewing')}: **{patient_label}**"
    )


# ---------------------------------------------------------------------------
# 9. Message styling
# ---------------------------------------------------------------------------


def render_assistant_message(content: str, kind: str | None = "answer") -> None:
    """Style an assistant message by response kind."""

    if kind == "denied":
        st.error(content, icon="🔒")
    elif kind == "no_data":
        st.info(content, icon="📭")
    elif kind == "blocked":
        st.warning(content, icon="🛡️")
    else:
        st.markdown(content)
