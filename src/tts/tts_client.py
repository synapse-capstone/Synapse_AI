from __future__ import annotations
import os
import hashlib
import time
from dotenv import load_dotenv
from google.cloud import texttospeech

load_dotenv()
GCP = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

_CACHE_DIR = ".cache_tts"
MAX_TTS_CHARS = 280  # 너무 긴 문장은 비용/시간 방지를 위해 자름


def _hash_key(text: str, lang: str, voice: str, speaking_rate: float, pitch: float) -> str:
    h = hashlib.md5()
    h.update("|".join([text, lang, voice, str(speaking_rate), str(pitch)]).encode("utf-8"))
    return h.hexdigest()


def _retry(fn, n: int = 2, delay: float = 0.6):
    """간단 재시도 래퍼 (지수 백오프)"""
    last = None
    for i in range(n + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if i < n:
                time.sleep(delay * (2**i))
    raise last


def synthesize(
    text: str,
    out_path: str = "response.mp3",  # 유지: 외부 시그니처 호환
    lang: str = "ko-KR",
    voice: str = "ko-KR-Standard-A",
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
) -> str:
    """
    입력 텍스트를 Google TTS로 MP3 생성.
    - 동일 파라미터/문구는 캐시 파일(.cache_tts/<md5>.mp3) 재사용.
    - 과도하게 긴 텍스트는 잘라서 합성.
    - 네트워크/일시 오류는 소규모 재시도.
    """
    if not text or not text.strip():
        raise ValueError("TTS text is empty.")

    if not (GCP and os.path.isfile(GCP)):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set to a valid JSON path")

    txt = text.strip()
    if len(txt) > MAX_TTS_CHARS:
        txt = txt[:MAX_TTS_CHARS] + "..."

    # 1) 캐시 조회
    os.makedirs(_CACHE_DIR, exist_ok=True)
    key = _hash_key(txt, lang, voice, speaking_rate, pitch)
    cached_path = os.path.abspath(os.path.join(_CACHE_DIR, f"{key}.mp3"))
    if os.path.exists(cached_path):
        return cached_path  # 캐시 적중

    # 2) 합성 (재시도 포함)
    client = texttospeech.TextToSpeechClient()
    ssml = texttospeech.SynthesisInput(text=txt)
    v = texttospeech.VoiceSelectionParams(language_code=lang, name=voice)
    cfg = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
        pitch=pitch,
    )

    resp = _retry(lambda: client.synthesize_speech(input=ssml, voice=v, audio_config=cfg))

    with open(cached_path, "wb") as f:
        f.write(resp.audio_content)

    return cached_path
