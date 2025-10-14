import os
from dotenv import load_dotenv

def check_stt_env():
    load_dotenv()
    ok = True
    issues = []

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        ok = False
        issues.append("Missing env: OPENAI_API_KEY")

    try:
        import openai  # noqa: F401
    except Exception as e:
        ok = False
        issues.append(f"openai import failed: {e}")

    return ok, issues

if __name__ == "__main__":
    ok, issues = check_stt_env()
    print("[STT] ✅ Ready" if ok else "[STT] ❌ Not ready")
    if issues: print(" -", "\n - ".join(issues))
