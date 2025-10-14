import os
from dotenv import load_dotenv

def check_tts_env():
    load_dotenv()
    ok = True
    issues = []

    try:
        from google.cloud import texttospeech  # noqa: F401
    except Exception as e:
        ok = False
        issues.append(f"google-cloud-texttospeech import failed: {e}")

    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        ok = False
        issues.append("Missing env: GOOGLE_APPLICATION_CREDENTIALS")
    elif not os.path.isfile(creds):
        ok = False
        issues.append(f"Key file not found at path: {creds}")

    return ok, issues

if __name__ == "__main__":
    ok, issues = check_tts_env()
    print("[TTS] ✅ Ready" if ok else "[TTS] ❌ Not ready")
    if issues: print(" -", "\n - ".join(issues))
