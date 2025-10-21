from __future__ import annotations
import argparse, os, subprocess, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.stt.whisper_client import transcribe_file
from src.pipeline.pipeline_mock import run_once
from src.tts.tts_client import synthesize

def play_mp3_mac(path: str):
    try:
        subprocess.run(["afplay", path], check=False)
    except Exception:
        print(f"(재생 실패) 파일 위치: {path}")

def run_file(audio_path: str, speak: bool = True) -> str:
    text = transcribe_file(audio_path, language="ko")
    print(f"[STT] {text}")
    response = run_once(text)
    print(f"[NLP] {response}")
    out_mp3 = synthesize(response, out_path="response.mp3", lang="ko-KR", voice="ko-KR-Standard-A")
    print(f"[TTS] 생성: {out_mp3}")
    if speak and sys.platform == "darwin":
        play_mp3_mac(out_mp3)
    return response

def main():
    ap = argparse.ArgumentParser(description="STT→NLP→TTS")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--no-speak", action="store_true")
    a = ap.parse_args()
    if not os.path.isfile(a.audio):
        print(f"파일 없음: {a.audio}", file=sys.stderr); sys.exit(2)
    run_file(a.audio, speak=not a.no_speak)

if __name__ == "__main__":
    main()
