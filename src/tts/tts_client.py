from __future__ import annotations
import os, hashlib
from dotenv import load_dotenv
from google.cloud import texttospeech

load_dotenv()
GCP = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

_CACHE_DIR = ".cache_tts"

def _hash_key(text: str, lang: str, voice: str, speaking_rate: float, pitch: float) -> str:
    h = hashlib.md5()
    h.update("|".join([text, lang, voice, str(speaking_rate), str(pitch)]).encode("utf-8"))
    return h.hexdigest()

def synthesize(
    text: str,
    out_path: str = "response.mp3",
    lang: str = "ko-KR",
    voice: str = "ko-KR-Standard-A",
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
) -> str:
    if not text or not text.strip():
        raise ValueError("TTS text is empty.")
    if not (GCP and os.path.isfile(GCP)):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set to a valid JSON path")

    # 1) 캐시 조회
    os.makedirs(_CACHE_DIR, exist_ok=True)
    key = _hash_key(text.strip(), lang, voice, speaking_rate, pitch)
    cached_path = os.path.abspath(os.path.join(_CACHE_DIR, f"{key}.mp3"))
    if os.path.exists(cached_path):
        # 요청한 out_path로 복사하지 않고, 캐시 경로 그대로 반환
        return cached_path

    # 2) 합성
    client = texttospeech.TextToSpeechClient()
    ssml = texttospeech.SynthesisInput(text=text)
    v = texttospeech.VoiceSelectionParams(language_code=lang, name=voice)
    cfg = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
        pitch=pitch,
    )
    resp = client.synthesize_speech(input=ssml, voice=v, audio_config=cfg)

    with open(cached_path, "wb") as f:
        f.write(resp.audio_content)

    return cached_path
