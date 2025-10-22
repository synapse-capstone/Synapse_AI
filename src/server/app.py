from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict
import tempfile, os, uuid

from src.dialogue.manager import DialogueCtx, next_turn
from src.stt.whisper_client import transcribe_file
from src.tts.tts_client import synthesize

app = FastAPI(title="Voice Kiosk API", version="1.0.0")

# 메모리 세션 저장소: 실제 운영은 Redis 등 외부 스토리지 추천
SESSIONS: Dict[str, DialogueCtx] = {}

class TextIn(BaseModel):
    session_id: str
    text: str

class StartOut(BaseModel):
    session_id: str
    response_text: str
    tts_path: str

def _ensure_session(session_id: str) -> DialogueCtx:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = DialogueCtx()
    return SESSIONS[session_id]

@app.post("/session/start", response_model=StartOut)
def session_start():
    session_id = uuid.uuid4().hex
    ctx = _ensure_session(session_id)
    # 첫 턴: BOOT -> GREETING
    resp_text = next_turn(ctx, "")
    tts_path = synthesize(resp_text, out_path=f"response_{session_id}.mp3")
    return StartOut(session_id=session_id, response_text=resp_text, tts_path=tts_path)

@app.post("/session/text")
def session_text(payload: TextIn):
    ctx = _ensure_session(payload.session_id)
    resp_text = next_turn(ctx, payload.text)
    tts_path = synthesize(resp_text, out_path=f"response_{payload.session_id}.mp3")
    return {"response_text": resp_text, "tts_path": tts_path}

@app.post("/session/voice")
async def session_voice(session_id: str, audio: UploadFile = File(...)):
    ctx = _ensure_session(session_id)
    # 업로드 파일을 임시 경로에 저장
    suffix = os.path.splitext(audio.filename or ".wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        user_text = transcribe_file(tmp_path, language="ko")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"STT 실패: {e}")
    finally:
        try: os.remove(tmp_path)
        except: pass

    resp_text = next_turn(ctx, user_text)
    tts_path = synthesize(resp_text, out_path=f"response_{session_id}.mp3")
    return {"stt_text": user_text, "response_text": resp_text, "tts_path": tts_path}

@app.get("/session/state")
def session_state(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="세션 없음")
    ctx = SESSIONS[session_id]
    # 간단 직렬화
    return {
        "state": ctx.state.name,
        "slots": ctx.slots,
        "cart": ctx.cart,
        "payment": ctx.payment
    }
