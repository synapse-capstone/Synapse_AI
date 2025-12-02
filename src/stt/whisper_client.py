from __future__ import annotations
import os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")  # proj_... (개인 키면 없어도 됨)

# 전역 Whisper 클라이언트 (서버 시작 시 미리 생성하여 재사용)
_whisper_client_cache = None

def _retry(fn, n=3, delay=1.0):
    err = None
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            err = e
            print(f"[Retry {i+1}/{n}] {e}")
            time.sleep(delay * (2**i))
    raise err

def _make_client():
    """Whisper 클라이언트 생성. 전역 캐시를 사용하여 재사용."""
    global _whisper_client_cache
    if _whisper_client_cache is not None:
        return _whisper_client_cache
    
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    if OPENAI_PROJECT:
        _whisper_client_cache = OpenAI(
            api_key=OPENAI_API_KEY,
            project=OPENAI_PROJECT,
            default_headers={"OpenAI-Project": OPENAI_PROJECT},
        )
    else:
        _whisper_client_cache = OpenAI(api_key=OPENAI_API_KEY)
    
    return _whisper_client_cache

def transcribe_file(path: str, language: str = "ko") -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio not found: {path}")
    client = _make_client()  # 전역 클라이언트 재사용
    def call():
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
            )
        return resp.text.strip()
    return _retry(call)
