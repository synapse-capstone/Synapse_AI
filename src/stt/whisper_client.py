from __future__ import annotations
import os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")  # proj_... (개인 키면 없어도 됨)

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
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    if OPENAI_PROJECT:
        return OpenAI(
            api_key=OPENAI_API_KEY,
            project=OPENAI_PROJECT,
            default_headers={"OpenAI-Project": OPENAI_PROJECT},
        )
    return OpenAI(api_key=OPENAI_API_KEY)

def transcribe_file(path: str, language: str = "ko") -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio not found: {path}")
    client = _make_client()
    def call():
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
            )
        return resp.text.strip()
    return _retry(call)
