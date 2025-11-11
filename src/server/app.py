from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict
import tempfile, os, uuid, time

from src.dialogue.manager import DialogueCtx, next_turn
from src.dialogue import prompts as P
from src.stt.whisper_client import transcribe_file
from src.tts.tts_client import synthesize
from src.pricing.price import load_configs

app = FastAPI(title="Voice Kiosk API", version="1.0.0")

# ── 세션/보안 가드 ──────────────────────────────────────────────────────────────
SESSIONS: Dict[str, DialogueCtx] = {}   # session_id -> context
SESS_META: Dict[str, float] = {}        # session_id -> last_active
SESSION_TTL = 600                       # 10분
MAX_TURNS = 20                          # 과도한 대화 방지
ACCEPTED_EXT = {".wav", ".mp3", ".m4a"} # 업로드 허용 포맷

# ── 모델 ──────────────────────────────────────────────────────────────────────
class TextIn(BaseModel):
    session_id: str
    text: str

class StartOut(BaseModel):
    session_id: str
    response_text: str
    tts_path: str

# ── 유틸 ──────────────────────────────────────────────────────────────────────
def _now() -> float:
    return time.time()

def _expired(ts: float) -> bool:
    return (_now() - ts) > SESSION_TTL

def _ensure_session(session_id: str | None = None) -> tuple[str, DialogueCtx]:
    if session_id and session_id in SESSIONS and not _expired(SESS_META.get(session_id, 0)):
        SESS_META[session_id] = _now()
        return session_id, SESSIONS[session_id]
    sid = session_id or uuid.uuid4().hex
    SESSIONS[sid] = DialogueCtx()
    SESS_META[sid] = _now()
    return sid, SESSIONS[sid]

def _reprompt_if_empty(text: str | None) -> str | None:
    """무음/짧은 발화 기본 처리: None이면 일반 플로우 진행."""
    if not text or len(text.strip()) < 2:
        return P.REPROMPT
    return None

def _maybe_close_if_too_long(sid: str, ctx: DialogueCtx):
    """턴 수가 MAX_TURNS를 넘으면 세션 종료 안내 후 초기화."""
    ctx.turns = getattr(ctx, "turns", 0) + 1
    if ctx.turns > MAX_TURNS:
        resp_text = "대화 시간이 길어져서 새로 시작할게요. 처음 화면으로 돌아갑니다."
        tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
        # 세션 정리
        SESSIONS.pop(sid, None)
        SESS_META.pop(sid, None)
        return {"response_text": resp_text, "tts_path": tts_path}
    return None

# ── 공개 엔드포인트 ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/version")
def version():
    return {"version": app.version, "stt": "openai-whisper-1", "tts": "google-tts"}

@app.get("/config/menu")
def config_menu():
    menu_cfg, opt_cfg = load_configs()
    return {"menus": menu_cfg, "options": opt_cfg}

# ── 세션/대화 ─────────────────────────────────────────────────────────────────
@app.post("/session/start", response_model=StartOut)
def session_start():
    sid, ctx = _ensure_session()
    # 첫 턴: BOOT -> GREETING
    resp_text = next_turn(ctx, "")
    tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
    return StartOut(session_id=sid, response_text=resp_text, tts_path=tts_path)

@app.post("/session/text")
def session_text(payload: TextIn):
    sid, ctx = _ensure_session(payload.session_id)

    # 무음/짧은 발화 처리
    maybe = _reprompt_if_empty(payload.text)
    if maybe:
        tts_path = synthesize(maybe, out_path=f"response_{sid}.mp3")
        return {"response_text": maybe, "tts_path": tts_path}

    # 턴 수 가드
    guard = _maybe_close_if_too_long(sid, ctx)
    if guard:
        return guard

    # 정상 처리
    resp_text = next_turn(ctx, payload.text)
    tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
    SESS_META[sid] = _now()
    return {"response_text": resp_text, "tts_path": tts_path}

@app.post("/session/voice")
async def session_voice(session_id: str, audio: UploadFile = File(...)):
    sid, ctx = _ensure_session(session_id)

    # 파일 확장자 검증
    suffix = os.path.splitext(audio.filename or ".wav")[1].lower()
    if suffix not in ACCEPTED_EXT:
        raise HTTPException(status_code=400, detail=f"허용되지 않은 형식: {suffix}")

    # 업로드 파일 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        user_text = transcribe_file(tmp_path, language="ko")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"STT 실패: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # 무음/짧은 발화 처리
    maybe = _reprompt_if_empty(user_text)
    if maybe:
        tts_path = synthesize(maybe, out_path=f"response_{sid}.mp3")
        return {"stt_text": user_text, "response_text": maybe, "tts_path": tts_path}

    # 턴 수 가드
    guard = _maybe_close_if_too_long(sid, ctx)
    if guard:
        return guard

    # 정상 처리
    resp_text = next_turn(ctx, user_text)
    tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
    SESS_META[sid] = _now()
    return {"stt_text": user_text, "response_text": resp_text, "tts_path": tts_path}

@app.get("/session/state")
def session_state(session_id: str):
    if session_id not in SESSIONS or _expired(SESS_META.get(session_id, 0)):
        raise HTTPException(status_code=404, detail="세션 없음")
    ctx = SESSIONS[session_id]
    SESS_META[session_id] = _now()
    return {
        "state": ctx.state.name,
        "slots": ctx.slots,
        "cart": ctx.cart,
        "payment": ctx.payment
    }
