import speech_recognition as sr
import spacy
from pydub import AudioSegment
import os
import re

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Supported audio formats
AUDIO_FORMATS = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]

# Speech-to-text function
def convert_audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()

    # Convert to .wav if it's not already
    ext = os.path.splitext(audio_file_path)[1].lower()
    if ext != ".wav":
        sound = AudioSegment.from_file(audio_file_path)
        audio_file_path = audio_file_path.rsplit(".", 1)[0] + ".wav"
        sound.export(audio_file_path, format="wav")

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Audio could not be understood."
        except sr.RequestError as e:
            return f"Could not request results; {e}"

# NLP extraction (issue, urgency, location)
def extract_complaint_details(transcribed_text):
    doc = nlp(transcribed_text.lower())

    # Define possible values
    issue_types = ["baggage", "delay", "cleaning", "security", "staff", "maintenance", "toilet", "gate"]
    urgency_levels = ["urgent", "immediate", "high", "low", "normal"]

    issue = None
    urgency = "normal"
    location = None

    # Match tokens with issue and urgency
    for token in doc:
        if not issue and token.lemma_ in issue_types:
            issue = token.lemma_
        if token.lemma_ in urgency_levels:
            urgency = token.lemma_

    # Named Entity Recognition for location
    for ent in doc.ents:
        if ent.label_ in ["GPE", "FACILITY", "LOC"]:
            location = ent.text.title()
            break

    # Fallback regex for gate/terminal pattern
    if not location:
        loc_match = re.search(r"(gate\s\d+|terminal\s\d+)", transcribed_text.lower())
        if loc_match:
            location = loc_match.group(0).title()

    return {
        "issue": issue or "general",
        "urgency": urgency,
        "location": location or "unknown",
        "raw_text": transcribed_text
    }
