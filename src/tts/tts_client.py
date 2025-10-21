from __future__ import annotations
import os
from dotenv import load_dotenv
from google.cloud import texttospeech

load_dotenv()
GCP = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def synthesize(
    text: str,
    out_path: str = "response.mp3",
    lang: str = "ko-KR",
    voice: str = "ko-KR-Standard-A",
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
) -> str:
    if not (GCP and os.path.isfile(GCP)):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set to a valid JSON path")
    client = texttospeech.TextToSpeechClient()
    ssml = texttospeech.SynthesisInput(text=text)
    v = texttospeech.VoiceSelectionParams(language_code=lang, name=voice)
    cfg = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
        pitch=pitch,
    )
    resp = client.synthesize_speech(input=ssml, voice=v, audio_config=cfg)
    with open(out_path, "wb") as f:
        f.write(resp.audio_content)
    return os.path.abspath(out_path)
