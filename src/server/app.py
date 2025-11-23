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
TTS_DIR = os.path.abspath(".cache_tts")  # 프로젝트 루트 기준
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
    BASE_URL은 배포 시 환경변수로 설정 권장.
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
        # dine_type -> menu_category -> menu_item -> temp/size -> options -> confirm -> payment -> done
        "step": "dine_type",
        "turns": 0,
        "dine_type": None,       # takeout / dinein
        "category": None,        # coffee / ade / tea / dessert
        "menu_id": None,         # COFFEE_AMERICANO ...
        "menu_name": None,       # "아메리카노" ...
        "temp": None,            # hot / ice
        "size": None,            # tall / grande / venti / ...
        "options": {
            "extra_shot": 0,     # 커피: 샷 추가
            "syrup": False,      # 커피: 시럽 추가 여부
            "decaf": None,       # 커피: True/False
            "sweetness": None,   # 에이드: low / normal / high
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
    '네', '응' 같은 한 글자 대답은 정상 입력으로 처리하기 위해
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
        "menu_id": ctx.get("menu_id"),
        "menu_name": ctx.get("menu_name"),
        "temp": ctx.get("temp"),
        "size": ctx.get("size"),
        "options": ctx.get("options"),
        "quantity": ctx.get("quantity"),
        "payment_method": ctx.get("payment_method"),
    }


# ── 파싱 유틸 ─────────────────────────────────────────────────────────────────
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
    if "에이드" in t or "음료" in t:
        return "ade"
    if "차" in t or "티" in t:
        return "tea"
    if "디저트" in t or "케이크" in t or "빵" in t:
        return "dessert"
    return None


def _menu_choices_for_category(cat: str) -> list[tuple[str, str]]:
    """카테고리별 (menu_id, menu_name) 리스트."""
    if cat == "coffee":
        return [
            ("COFFEE_AMERICANO", "아메리카노"),
            ("COFFEE_ESPRESSO", "에스프레소"),
            ("COFFEE_LATTE", "카페 라떼"),
            ("COFFEE_CAPPUCCINO", "카푸치노"),
        ]
    if cat == "ade":
        return [
            ("ADE_LEMON", "레몬에이드"),
            ("ADE_GRAPEFRUIT", "자몽에이드"),
            ("ADE_GREEN_GRAPE", "청포도 에이드"),
            ("ADE_ORANGE", "오렌지 에이드"),
        ]
    if cat == "tea":
        return [
            ("TEA_CHAMOMILE", "캐모마일 티"),
            ("TEA_EARL_GREY", "얼그레이 티"),
            ("TEA_YUJA", "유자차"),
            ("TEA_GREEN", "녹차"),
        ]
    if cat == "dessert":
        return [
            ("DESSERT_CHEESECAKE", "치즈케이크"),
            ("DESSERT_TIRAMISU", "티라미수"),
            ("DESSERT_BROWNIE", "초코 브라우니"),
            ("DESSERT_CROISSANT", "크루아상"),
        ]
    return []


def _parse_menu_item(category: str | None, text: str) -> tuple[str, str] | None:
    """사용자 발화에서 메뉴를 찾아 (menu_id, menu_name) 반환."""
    if not category:
        return None
    t = text.replace(" ", "").lower()
    for mid, name in _menu_choices_for_category(category):
        key = name.replace(" ", "").lower()
        if key in t:
            return mid, name

    # 약간의 별칭 처리
    if category == "coffee":
        if "아메" in t:
            return "COFFEE_AMERICANO", "아메리카노"
        if "라떼" in t:
            return "COFFEE_LATTE", "카페 라떼"
        if "카푸" in t:
            return "COFFEE_CAPPUCCINO", "카푸치노"
    if category == "ade":
        if "레몬" in t:
            return "ADE_LEMON", "레몬에이드"
        if "자몽" in t:
            return "ADE_GRAPEFRUIT", "자몽에이드"
        if "청포도" in t:
            return "ADE_GREEN_GRAPE", "청포도 에이드"
        if "오렌지" in t:
            return "ADE_ORANGE", "오렌지 에이드"
    if category == "tea":
        if "캐모" in t:
            return "TEA_CHAMOMILE", "캐모마일 티"
        if "얼그" in t:
            return "TEA_EARL_GREY", "얼그레이 티"
        if "유자" in t:
            return "TEA_YUJA", "유자차"
        if "녹차" in t:
            return "TEA_GREEN", "녹차"
    if category == "dessert":
        if "치즈케" in t or "치즈" in t:
            return "DESSERT_CHEESECAKE", "치즈케이크"
        if "티라" in t:
            return "DESSERT_TIRAMISU", "티라미수"
        if "브라우니" in t or "브라우" in t:
            return "DESSERT_BROWNIE", "초코 브라우니"
        if "크루아상" in t or "크로와상" in t:
            return "DESSERT_CROISSANT", "크루아상"
    return None


def _parse_temp(text: str) -> str | None:
    t = text.replace(" ", "").lower()
    if "아이스" in t or "ice" in t or "차갑" in t:
        return "ice"
    if "핫" in t or "hot" in t or "따뜻" in t or "뜨거" in t or "뜨겁" in t:
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


def _parse_options(category: str | None, text: str, options: dict) -> dict:
    """카테고리별 옵션 파싱."""
    t = text.replace(" ", "").lower()

    if category == "coffee":
        # 디카페인
        if "디카페인" in t or "디카" in t:
            options["decaf"] = True
        elif "일반" in t and "카페인" in t:
            options["decaf"] = False

        # 시럽
        if "시럽" in t and ("추가" in t or "넣어" in t):
            options["syrup"] = True

        # 샷 추가
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

    elif category == "ade":
        # 당도: 연하게/보통/달게
        if any(x in t for x in ["연하게", "적게", "연한", "조금달게"]):
            options["sweetness"] = "low"
        elif any(x in t for x in ["보통", "기본", "그냥"]):
            options["sweetness"] = "normal"
        elif any(x in t for x in ["달게", "많이달게", "달달", "많이달"]):
            options["sweetness"] = "high"

    # tea, dessert는 별도 옵션 없음 (size/temp만 사용)
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
    if any(x in t for x in ["네", "응", "예", "맞아요", "맞아", "좋아요", "그래"]):
        return "yes"
    if any(x in t for x in ["아니", "아니요", "싫", "다시"]):
        return "no"
    return None


# ── backend_payload 생성 ─────────────────────────────────────────────────────
def _build_backend_payload(ctx: Dict[str, Any]) -> dict | None:
    """
    현재까지의 선택을 기반으로 백엔드에 넘길 주문 JSON 예시 생성.
    """
    category = ctx.get("category")
    temp = ctx.get("temp")
    size = ctx.get("size")
    quantity = ctx.get("quantity", 1)
    options = ctx.get("options", {}) or {}
    dine_type = ctx.get("dine_type")
    payment_method = ctx.get("payment_method")

    if not category and not ctx.get("menu_id"):
        return None

    menu_id = ctx.get("menu_id")
    menu_name = ctx.get("menu_name")

    # menu_id/menu_name이 아직 없으면 카테고리 디폴트로 세팅
    if not menu_id or not menu_name:
        if category == "coffee":
            menu_id = "COFFEE_DEFAULT"
            menu_name = "커피"
        elif category == "ade":
            menu_id = "ADE_DEFAULT"
            menu_name = "에이드"
        elif category == "tea":
            menu_id = "TEA_DEFAULT"
            menu_name = "차"
        elif category == "dessert":
            menu_id = "DESSERT_DEFAULT"
            menu_name = "디저트"

    return {
        "category": category,
        "menu_id": menu_id,
        "menu_name": menu_name,
        "temp": temp,
        "size": size,
        "quantity": quantity,
        "base_price": None,  # 가격은 pricing 모듈과 연동되면 채우기
        "options": {
            "extra_shot": options.get("extra_shot", 0),
            "syrup": options.get("syrup", False),
            "decaf": options.get("decaf"),
            "sweetness": options.get("sweetness"),
        },
        "dine_type": dine_type,
        "payment_method": payment_method,
    }


# ── 주문 요약 문장 생성 ──────────────────────────────────────────────────────
def _order_summary_sentence(ctx: Dict[str, Any]) -> str:
    category = ctx.get("category")
    menu_name = ctx.get("menu_name") or {
        "coffee": "커피",
        "ade": "에이드",
        "tea": "차",
        "dessert": "디저트",
    }.get(category, "메뉴")

    temp = ctx.get("temp")
    size = ctx.get("size")
    qty = ctx.get("quantity", 1)
    options = ctx.get("options", {}) or {}

    temp_str = ""
    if temp == "ice":
        temp_str = "아이스 "
    elif temp == "hot":
        temp_str = "뜨거운 "

    size_str = {
        "tall": "톨 ",
        "grande": "그란데 ",
        "venti": "벤티 ",
        "small": "스몰 ",
        "medium": "미디엄 ",
        "large": "라지 ",
    }.get(size, "")

    # 잔/개 단위
    if category in ("coffee", "ade", "tea"):
        unit = "잔"
    else:
        unit = "개"

    opt_parts: list[str] = []

    if category == "coffee":
        if options.get("decaf"):
            opt_parts.append("디카페인")
        if options.get("extra_shot", 0) > 0:
            opt_parts.append(f"샷 {options['extra_shot']}번 추가")
        if options.get("syrup"):
            opt_parts.append("시럽 추가")
    elif category == "ade":
        sweetness = options.get("sweetness")
        if sweetness == "low":
            opt_parts.append("당도 낮게")
        elif sweetness == "normal":
            opt_parts.append("당도 보통")
        elif sweetness == "high":
            opt_parts.append("당도 높게")

    opt_str = ", ".join(opt_parts) if opt_parts else "옵션 없이"

    return f"{temp_str}{size_str}{menu_name} {qty}{unit}, {opt_str}로 주문하실 건가요?"


# ── 메인 대화 흐름 로직 ──────────────────────────────────────────────────────
def _handle_turn(ctx: Dict[str, Any], user_text: str) -> str:
    """
    한 턴의 입력(user_text)에 대해,
    ctx 상태를 업데이트하고, 사용자에게 들려줄 response_text를 반환.
    """
    text = (user_text or "").strip()
    step = ctx.get("step", "dine_type")
    category = ctx.get("category")

    # 1) 먹고가기 / 들고가기
    if step == "dine_type":
        dine = _parse_dine_type(text)
        if dine is None:
            ctx["step"] = "dine_type"
            return "포장해서 가져가시나요, 매장에서 드시나요?"
        ctx["dine_type"] = dine
        ctx["step"] = "menu_category"
        where = "포장" if dine == "takeout" else "매장에서 식사"
        return f"{where}로 진행할게요. 커피, 에이드, 차, 디저트 중 무엇을 드시겠어요?"

    # 2) 메뉴 대분류 선택 (커피/에이드/차/디저트)
    if step == "menu_category":
        cat = _parse_category(text)
        if cat is None:
            return "커피, 에이드, 차, 디저트 중에서 한 가지만 말씀해 주세요."
        ctx["category"] = cat
        ctx["menu_id"] = None
        ctx["menu_name"] = None
        ctx["temp"] = None
        ctx["size"] = None
        ctx["options"] = {
            "extra_shot": 0,
            "syrup": False,
            "decaf": None,
            "sweetness": None,
        }
        ctx["step"] = "menu_item"

        choices = _menu_choices_for_category(cat)
        if cat == "coffee":
            cat_name = "커피 메뉴"
        elif cat == "ade":
            cat_name = "에이드 메뉴"
        elif cat == "tea":
            cat_name = "티 메뉴"
        else:
            cat_name = "디저트 메뉴"
        names = " / ".join(name for _, name in choices)
        return f"{cat_name}에서 어떤 걸 드시겠어요? 예: {names}"

    # 3) 세부 메뉴 선택 (아메리카노, 레몬에이드, 치즈케이크 등)
    if step == "menu_item":
        parsed = _parse_menu_item(category, text)
        if not parsed:
            choices = _menu_choices_for_category(category)
            names = " / ".join(name for _, name in choices)
            return f"죄송해요, 잘 못 들었어요. 다시 한 번 메뉴를 말씀해 주세요. 예: {names}"
        menu_id, menu_name = parsed
        ctx["menu_id"] = menu_id
        ctx["menu_name"] = menu_name

        # 카테고리별로 다음 단계 분기
        if category in ("coffee", "tea"):
            ctx["step"] = "temp"
            return f"{menu_name}를 선택하셨어요. 따뜻하게 드실까요, 아이스로 드실까요?"
        if category == "ade":
            ctx["step"] = "size"
            return f"{menu_name}를 선택하셨어요. 사이즈는 톨, 그란데, 벤티 중에서 선택해 주세요."
        if category == "dessert":
            ctx["step"] = "confirm"
            return _order_summary_sentence(ctx)

    # 4) 온도 선택 (커피/차)
    if step == "temp":
        temp = _parse_temp(text)
        if temp is None:
            return "따뜻하게 드실지, 아이스로 드실지 말씀해 주세요. 예: '아이스로 주세요'."
        ctx["temp"] = temp
        ctx["step"] = "size"
        how = "아이스" if temp == "ice" else "뜨겁게"
        return f"{how}로 준비할게요. 사이즈는 톨, 그란데, 벤티 중에서 선택해 주세요."

    # 5) 사이즈 선택
    if step == "size":
        size = _parse_size(text)
        if size is None:
            return "사이즈를 다시 말씀해 주세요. 톨, 그란데, 벤티 중 하나를 선택해 주세요."
        ctx["size"] = size

        if category == "coffee":
            ctx["step"] = "options"
            return "옵션을 선택해 주세요. 디카페인 여부, 샷 추가, 시럽 추가가 필요하면 말씀해 주세요."
        if category == "ade":
            ctx["step"] = "options"
            return "당도는 연하게, 보통, 달게 중에서 어떻게 해 드릴까요?"
        if category == "tea":
            ctx["step"] = "confirm"
            return _order_summary_sentence(ctx)
        if category == "dessert":
            ctx["step"] = "confirm"
            return _order_summary_sentence(ctx)

    # 6) 옵션 선택 (커피/에이드)
    if step == "options":
        options = ctx.get("options", {})
        ctx["options"] = _parse_options(category, text, options)
        ctx["step"] = "confirm"
        return _order_summary_sentence(ctx)

    # 7) 주문 내역 확인
    if step == "confirm":
        yn = _yes_no(text)
        if yn == "yes":
            ctx["step"] = "payment"
            return "결제 수단을 선택해 주세요. 카드, 현금, 카카오페이 등으로 말씀해 주세요."
        if yn == "no":
            # 메뉴 종류부터 다시
            ctx.update(_new_session_ctx())
            ctx["step"] = "menu_category"
            return "알겠습니다. 다시 메뉴 종류부터 선택할게요. 커피, 에이드, 차, 디저트 중에서 말씀해 주세요."
        return "주문이 맞으면 '네', 다시 선택하시려면 '아니요'라고 말씀해 주세요."

    # 8) 결제 수단 선택
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

    # 9) 주문 완료 후 새 주문
    if step == "done":
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

    # 무음 처리
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
        "stt_text": payload.text,
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

    # 무음 처리
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
