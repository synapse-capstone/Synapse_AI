from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
import tempfile, os, uuid, time, re

from src.stt.whisper_client import transcribe_file
from src.tts.tts_client import synthesize
from src.pricing.price import load_configs

app = FastAPI(title="Voice Kiosk API", version="1.0.0")

# ── 세션/보안 가드 ──────────────────────────────────────────────────────────────
SESSIONS: Dict[str, Dict[str, Any]] = {}   # session_id -> context(dict)
SESS_META: Dict[str, float] = {}           # session_id -> last_active
SESSION_TTL = 600                          # 10분
MAX_TURNS = 20                             # 과도한 대화 방지
ACCEPTED_EXT = {".wav", ".mp3", ".m4a"}    # 업로드 허용 포맷

# ── TTS 파일 제공 관련 ─────────────────────────────────────────────────────────
TTS_DIR = os.path.abspath(".cache_tts")  # tts_client.py와 동일 디렉터리
_TTS_NAME_RE = re.compile(r"^[a-f0-9]{32}\.mp3$", re.IGNORECASE)


def _tts_path_from_name(name: str) -> str:
    """파일명 검증 + 디렉터리 탈출 방지."""
    if not _TTS_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="잘못된 파일명 형식")
    abs_path = os.path.abspath(os.path.join(TTS_DIR, name))
    if not abs_path.startswith(TTS_DIR + os.sep):
        raise HTTPException(status_code=400, detail="경로가 올바르지 않습니다")
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
    return abs_path


def _make_tts_url(tts_path: str) -> str:
    """
    응답에 절대 URL을 포함하고 싶을 때 사용.
    BASE_URL은 배포 시 환경변수로 설정 권장 (예: https://voice-kiosk.example.com)
    개발 기본값은 로컬 서버.
    """
    base = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    fname = os.path.basename(tts_path)
    if not _TTS_NAME_RE.match(fname):
        return ""
    return f"{base}/tts/{fname}"


# ── 모델 ──────────────────────────────────────────────────────────────────────
class TextIn(BaseModel):
    session_id: str
    text: str


class StartOut(BaseModel):
    session_id: str
    response_text: str
    tts_path: str
    tts_url: str | None = None
    context: dict | None = None
    backend_payload: dict | None = None


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def _now() -> float:
    return time.time()


def _expired(ts: float) -> bool:
    return (_now() - ts) > SESSION_TTL


def _new_session_ctx() -> Dict[str, Any]:
    """새 세션 기본 상태."""
    return {
        # 대화 단계:
        # dine_type -> menu_category -> temp -> size -> options -> confirm -> payment -> done
        "step": "dine_type",
        "turns": 0,
        "dine_type": None,       # takeout / dinein
        "category": None,        # coffee / tea / drink / snack
        "temp": None,            # hot / ice
        "size": None,            # tall / grande / venti / small / medium / large
        "options": {
            "caffeine": None,    # regular / decaf
            "syrup": False,      # True/False
            "whip": False,       # True/False
            "extra_shot": 0,     # int
        },
        "quantity": 1,
        "payment_method": None,  # card / cash / kakaopay / ...
    }


def _ensure_session(session_id: str | None = None) -> tuple[str, Dict[str, Any]]:
    if session_id and session_id in SESSIONS and not _expired(SESS_META.get(session_id, 0)):
        ctx = SESSIONS[session_id]
    else:
        session_id = session_id or uuid.uuid4().hex
        ctx = _new_session_ctx()
        SESSIONS[session_id] = ctx
    SESS_META[session_id] = _now()
    return session_id, ctx


def _reprompt_if_empty(text: str | None) -> str | None:
    """
    무음/빈 문자열 기본 처리.
    ※ '네', '응' 같은 한 글자 대답은 정상 입력으로 처리하기 위해
       길이 체크는 하지 않고, 진짜 공백/None일 때만 재질문.
    """
    if text is None or not text.strip():
        return "죄송해요, 잘 못 들었어요. 다시 한번 말씀해 주시겠어요?"
    return None


def _maybe_close_if_too_long(sid: str, ctx: Dict[str, Any]):
    """턴 수가 MAX_TURNS를 넘으면 세션 종료 안내 후 초기화."""
    ctx["turns"] = ctx.get("turns", 0) + 1
    if ctx["turns"] > MAX_TURNS:
        resp_text = "대화 시간이 길어져서 새로 시작할게요. 처음 화면으로 돌아갑니다."
        tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
        # 세션 정리
        SESSIONS.pop(sid, None)
        SESS_META.pop(sid, None)
        return {
            "response_text": resp_text,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": None,
            "backend_payload": None,
        }
    return None


def _ctx_snapshot(ctx: Dict[str, Any]) -> dict:
    """프론트/백엔드 참고용 현재 상태 스냅샷."""
    return {
        "step": ctx.get("step"),
        "dine_type": ctx.get("dine_type"),
        "category": ctx.get("category"),
        "temp": ctx.get("temp"),
        "size": ctx.get("size"),
        "options": ctx.get("options"),
        "quantity": ctx.get("quantity"),
        "payment_method": ctx.get("payment_method"),
    }


# ── 간단한 규칙 기반 파싱들 ──────────────────────────────────────────────────
def _parse_dine_type(text: str) -> str | None:
    t = text.replace(" ", "")
    if "포장" in t or "들고갈" in t or "가져갈" in t:
        return "takeout"
    if "먹고갈" in t or "매장" in t or "여기서" in t:
        return "dinein"
    return None


def _parse_category(text: str) -> str | None:
    t = text.replace(" ", "")
    if "커피" in t:
        return "coffee"
    if "차" in t:
        return "tea"
    if "음료" in t or "주스" in t:
        return "drink"
    if "간식" in t or "디저트" in t or "케이크" in t:
        return "snack"
    return None


def _parse_temp(text: str) -> str | None:
    t = text.replace(" ", "").lower()
    if "아이스" in t or "ice" in t or "차갑" in t:
        return "ice"
    if "핫" in t or "hot" in t or "따뜻" in t or "뜨거" in t:
        return "hot"
    return None


def _parse_size(text: str) -> str | None:
    t = text.replace(" ", "").lower()
    if "톨" in t or "tall" in t:
        return "tall"
    if "그란데" in t or "grande" in t:
        return "grande"
    if "벤티" in t or "venti" in t:
        return "venti"
    if "스몰" in t or "small" in t or "작은" in t:
        return "small"
    if "미디엄" in t or "medium" in t or "중간" in t:
        return "medium"
    if "라지" in t or "large" in t or "큰" in t:
        return "large"
    return None


def _parse_options(text: str, options: dict) -> dict:
    t = text.replace(" ", "").lower()

    # 디카페인
    if "디카페인" in t or "디카" in t:
        options["caffeine"] = "decaf"
    elif "카페인" in t and ("일반" in t or "보통" in t):
        options["caffeine"] = "regular"

    # 시럽
    if "시럽" in t and ("추가" in t or "넣어" in t):
        options["syrup"] = True

    # 휘핑
    if "휘핑" in t and ("추가" in t or "넣어" in t or "올려" in t):
        options["whip"] = True

    # 샷 추가 (한/두/세 + 숫자)
    extra = options.get("extra_shot", 0)
    if "샷" in t:
        if "두" in t or "2" in t:
            extra = 2
        elif "세" in t or "3" in t:
            extra = 3
        elif "한" in t or "1" in t:
            extra = 1
        else:
            extra = max(extra, 1)
    options["extra_shot"] = extra

    return options


def _parse_payment(text: str) -> str | None:
    t = text.replace(" ", "").lower()
    if "카드" in t:
        return "card"
    if "현금" in t:
        return "cash"
    if "카카오페이" in t or "카톡페이" in t:
        return "kakaopay"
    if "삼성페이" in t:
        return "samsungpay"
    if "페이" in t:
        return "pay"
    return None


def _yes_no(text: str) -> str | None:
    t = text.replace(" ", "")
    if any(x in t for x in ["네", "응", "예", "맞아", "좋아", "네네", "그래"]):
        return "yes"
    if any(x in t for x in ["아니", "아니요", "싫", "다시"]):
        return "no"
    return None


# ── backend_payload 생성 ─────────────────────────────────────────────────────
def _build_backend_payload(ctx: Dict[str, Any]) -> dict | None:
    """
    현재까지의 선택을 기반으로 백엔드에 넘길 주문 JSON 예시 생성.
    - 카테고리/온도/사이즈가 어느 정도 정해졌을 때부터 채움.
    """
    category = ctx.get("category")
    temp = ctx.get("temp")
    size = ctx.get("size")
    quantity = ctx.get("quantity", 1)
    options = ctx.get("options", {}) or {}

    if not category and not temp and not size:
        return None

    # 메뉴 이름은 임시로 카테고리 기준 대표 메뉴로 매핑
    menu_id = None
    menu_name = None
    if category == "coffee":
        menu_id = "COFFEE_DEFAULT"
        menu_name = "커피"
    elif category == "tea":
        menu_id = "TEA_DEFAULT"
        menu_name = "차"
    elif category == "drink":
        menu_id = "DRINK_DEFAULT"
        menu_name = "음료"
    elif category == "snack":
        menu_id = "SNACK_DEFAULT"
        menu_name = "간식"

    return {
        "category": category,
        "menu_id": menu_id,
        "menu_name": menu_name,
        "temp": temp,
        "size": size,
        "quantity": quantity,
        "base_price": None,  # 가격은 pricing 쪽과 연동되면 채우기
        "options": {
            "caffeine": options.get("caffeine"),
            "syrup": options.get("syrup"),
            "whip": options.get("whip"),
            "extra_shot": options.get("extra_shot", 0),
        },
        "dine_type": ctx.get("dine_type"),
        "payment_method": ctx.get("payment_method"),
    }


# ── 메인 대화 흐름 로직 ──────────────────────────────────────────────────────
def _handle_turn(ctx: Dict[str, Any], user_text: str) -> str:
    """
    한 턴의 입력(user_text)에 대해,
    ctx 상태를 업데이트하고, 사용자에게 들려줄 response_text를 반환.
    """
    text = (user_text or "").strip()
    step = ctx.get("step", "dine_type")

    # 1) 먹고가기 / 들고가기
    if step == "dine_type":
        dine = _parse_dine_type(text)
        if dine is None:
            ctx["step"] = "dine_type"
            return "포장해서 가져가시나요, 매장에서 드시나요?"
        ctx["dine_type"] = dine
        ctx["step"] = "menu_category"
        where = "포장" if dine == "takeout" else "매장에서 식사"
        return f"{where}로 진행할게요. 커피, 차, 음료, 간식 중 무엇을 드시겠어요?"

    # 2) 메뉴 종류 선택 (커피/차/음료/간식)
    if step == "menu_category":
        cat = _parse_category(text)
        if cat is None:
            return "커피, 차, 음료, 간식 중에서 한 가지만 말씀해 주세요."
        ctx["category"] = cat
        ctx["step"] = "temp"
        spoken = {"coffee": "커피", "tea": "차", "drink": "음료", "snack": "간식"}[cat]
        return f"{spoken}를 선택하셨어요. 따뜻하게 드실까요, 아이스로 드실까요?"

    # 3) 온도 선택
    if step == "temp":
        temp = _parse_temp(text)
        if temp is None:
            return "따뜻하게 드실지, 아이스로 드실지 말씀해 주세요. 예: '아이스로 주세요'."
        ctx["temp"] = temp
        ctx["step"] = "size"
        how = "아이스" if temp == "ice" else "뜨겁게"
        return f"{how}로 준비할게요. 사이즈는 톨, 그란데, 벤티 중에서 선택해 주세요."

    # 4) 사이즈 선택
    if step == "size":
        size = _parse_size(text)
        if size is None:
            return "사이즈를 다시 말씀해 주세요. 톨, 그란데, 벤티 중 하나를 선택해 주세요."
        ctx["size"] = size
        ctx["step"] = "options"
        return "옵션을 선택해 주세요. 디카페인 여부, 샷 추가, 시럽 추가, 휘핑 추가가 필요하면 말씀해 주세요."

    # 5) 옵션 선택
    if step == "options":
        options = ctx.get("options", {})
        ctx["options"] = _parse_options(text, options)
        ctx["step"] = "confirm"

        cat = ctx.get("category")
        temp = ctx.get("temp")
        size = ctx.get("size")
        opt = ctx["options"]

        spoken_cat = {"coffee": "커피", "tea": "차", "drink": "음료", "snack": "간식"}.get(cat, "메뉴")
        spoken_temp = "아이스" if temp == "ice" else ("뜨거운" if temp == "hot" else "")
        spoken_size = {
            "tall": "톨",
            "grande": "그란데",
            "venti": "벤티",
            "small": "스몰",
            "medium": "미디엄",
            "large": "라지",
        }.get(size, "")

        opt_parts = []
        if opt.get("caffeine") == "decaf":
            opt_parts.append("디카페인")
        if opt.get("extra_shot", 0) > 0:
            opt_parts.append(f"샷 {opt['extra_shot']}번 추가")
        if opt.get("syrup"):
            opt_parts.append("시럽 추가")
        if opt.get("whip"):
            opt_parts.append("휘핑 추가")

        opt_str = ", ".join(opt_parts) if opt_parts else "옵션 없이"
        return f"{spoken_temp} {spoken_size} {spoken_cat} 한 잔, {opt_str}로 주문하실 건가요?"

    # 6) 주문 내역 확인
    if step == "confirm":
        yn = _yes_no(text)
        if yn == "yes":
            ctx["step"] = "payment"
            return "결제 수단을 선택해 주세요. 카드, 현금, 카카오페이 등으로 말씀해 주세요."
        if yn == "no":
            # 간단하게 처음 메뉴 선택으로 되돌리기
            ctx["category"] = None
            ctx["temp"] = None
            ctx["size"] = None
            ctx["options"] = {
                "caffeine": None,
                "syrup": False,
                "whip": False,
                "extra_shot": 0,
            }
            ctx["step"] = "menu_category"
            return "알겠습니다. 다시 메뉴 종류부터 선택할게요. 커피, 차, 음료, 간식 중에서 말씀해 주세요."
        return "주문이 맞으면 '네', 다시 선택하시려면 '아니요'라고 말씀해 주세요."

    # 7) 결제 수단 선택
    if step == "payment":
        pay = _parse_payment(text)
        if pay is None:
            return "결제 수단을 다시 말씀해 주세요. 카드, 현금, 카카오페이 등으로 말씀해 주세요."
        ctx["payment_method"] = pay
        ctx["step"] = "done"
        spoken_pay = {
            "card": "카드",
            "cash": "현금",
            "kakaopay": "카카오페이",
            "samsungpay": "삼성페이",
            "pay": "간편결제",
        }.get(pay, "선택하신 결제 수단")
        return f"{spoken_pay}로 결제 도와드릴게요. 주문이 완료되었습니다. 감사합니다."

    # 8) 주문 완료 후
    if step == "done":
        # 새 주문으로 리셋
        ctx.clear()
        ctx.update(_new_session_ctx())
        return "새 주문을 도와드릴게요. 포장해서 가져가시나요, 매장에서 드시나요?"

    # 안전장치: 알 수 없는 상태면 처음으로
    ctx.clear()
    ctx.update(_new_session_ctx())
    return "처음부터 다시 진행할게요. 포장해서 가져가시나요, 매장에서 드시나요?"


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
    # 첫 턴: 먹고가기/들고가기 물어보기
    resp_text = "포장해서 가져가시나요, 매장에서 드시나요?"
    tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
    backend_payload = _build_backend_payload(ctx)
    return {
        "session_id": sid,
        "response_text": resp_text,
        "tts_path": tts_path,
        "tts_url": _make_tts_url(tts_path) or None,
        "context": _ctx_snapshot(ctx),
        "backend_payload": backend_payload,
    }


@app.post("/session/text")
def session_text(payload: TextIn):
    sid, ctx = _ensure_session(payload.session_id)

    # 무음/짧은 발화 처리
    maybe = _reprompt_if_empty(payload.text)
    if maybe:
        tts_path = synthesize(maybe, out_path=f"response_{sid}.mp3")
        backend_payload = _build_backend_payload(ctx)
        return {
            "stt_text": payload.text,
            "response_text": maybe,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": backend_payload,
        }

    # 턴 수 가드
    guard = _maybe_close_if_too_long(sid, ctx)
    if guard:
        return guard

    # 정상 처리
    resp_text = _handle_turn(ctx, payload.text)
    tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
    SESS_META[sid] = _now()
    backend_payload = _build_backend_payload(ctx)
    return {
        "stt_text": payload.text,  # 텍스트 모드에서도 프론트가 보기 좋게 그대로 실어줌
        "response_text": resp_text,
        "tts_path": tts_path,
        "tts_url": _make_tts_url(tts_path) or None,
        "context": _ctx_snapshot(ctx),
        "backend_payload": backend_payload,
    }


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
        backend_payload = _build_backend_payload(ctx)
        return {
            "stt_text": user_text,
            "response_text": maybe,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": backend_payload,
        }

    # 턴 수 가드
    guard = _maybe_close_if_too_long(sid, ctx)
    if guard:
        return guard

    # 정상 처리
    resp_text = _handle_turn(ctx, user_text)
    tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
    SESS_META[sid] = _now()
    backend_payload = _build_backend_payload(ctx)
    return {
        "stt_text": user_text,   # STT 결과
        "response_text": resp_text,
        "tts_path": tts_path,
        "tts_url": _make_tts_url(tts_path) or None,
        "context": _ctx_snapshot(ctx),
        "backend_payload": backend_payload,
    }


@app.get("/session/state")
def session_state(session_id: str):
    if session_id not in SESSIONS or _expired(SESS_META.get(session_id, 0)):
        raise HTTPException(status_code=404, detail="세션 없음")
    ctx = SESSIONS[session_id]
    SESS_META[session_id] = _now()
    return _ctx_snapshot(ctx)


# ── TTS 파일 다운로드/스트리밍 ────────────────────────────────────────────────
@app.get("/tts/{filename}")
def get_tts_file(filename: str):
    """
    생성된 TTS mp3를 내려주는 엔드포인트.
    - filename: 예) 'd96d9e768275ce350fb49bdf3f248ab1.mp3'
    """
    path = _tts_path_from_name(filename)
    return FileResponse(path, media_type="audio/mpeg", filename=filename)
