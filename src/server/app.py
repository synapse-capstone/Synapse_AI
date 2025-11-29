from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
import tempfile, os, uuid, time, re, json
from openai import OpenAI

from src.stt.whisper_client import transcribe_file
from src.tts.tts_client import synthesize
from src.pricing.price import load_configs

app = FastAPI(title="Voice Kiosk API", version="1.0.0")

# OpenAI 클라이언트 (환경변수 OPENAI_API_KEY 사용)
gpt_client = OpenAI()

# ───────────────────────────────────────────────
# 세션 관리
# ───────────────────────────────────────────────
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESS_META: Dict[str, float] = {}
SESSION_TTL = 600       # 10분
MAX_TURNS = 20
ACCEPTED_EXT = {".wav", ".mp3", ".m4a"}

# TTS 파일
TTS_DIR = os.path.abspath(".cache_tts")
_TTS_NAME_RE = re.compile(r"^[a-f0-9]{32}\.mp3$", re.IGNORECASE)


def _tts_path_from_name(name: str) -> str:
    """TTS 캐시 파일 이름 검증 및 경로 확보."""
    if not _TTS_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="잘못된 파일명 형식입니다.")

    abs_path = os.path.abspath(os.path.join(TTS_DIR, name))

    if not abs_path.startswith(TTS_DIR + os.sep):
        raise HTTPException(status_code=400, detail="경로가 유효하지 않습니다.")

    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    return abs_path


def _make_tts_url(tts_path: str) -> str:
    """프론트에서 재생할 수 있는 절대 URL 생성."""
    base = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    fname = os.path.basename(tts_path)
    if not _TTS_NAME_RE.match(fname):
        return ""
    return f"{base}/tts/{fname}"


# ───────────────────────────────────────────────
# Pydantic Models
# ───────────────────────────────────────────────
class TextIn(BaseModel):
    session_id: str
    text: str
    is_help: bool = False  # 프론트에서 "도움말 모드" 플래그를 줄 수도 있음 (안 써도 됨)


class StartOut(BaseModel):
    session_id: str
    response_text: str
    tts_path: str
    tts_url: str | None = None
    context: dict | None = None
    backend_payload: dict | None = None


# ───────────────────────────────────────────────
# 유틸
# ───────────────────────────────────────────────
def _now() -> float:
    return time.time()


def _expired(ts: float) -> bool:
    return (_now() - ts) > SESSION_TTL


def _new_session_ctx() -> Dict[str, Any]:
    """새 세션 기본 상태."""
    return {
        # dine_type -> menu_category -> menu_item -> temp -> size -> options -> confirm -> payment -> done
        "step": "dine_type",
        "turns": 0,
        "dine_type": None,        # takeout / dinein
        "category": None,         # coffee / ade / tea / dessert
        "menu_id": None,
        "menu_name": None,
        "temp": None,             # hot / ice
        "size": None,             # tall / grande / venti / ...
        "options": {
            "extra_shot": 0,      # 커피: 샷 추가
            "syrup": False,       # 커피: 시럽 추가
            "decaf": None,        # 커피: 디카페인 여부
            "sweetness": None,    # 에이드: low / normal / high
        },
        "quantity": 1,
        "payment_method": None,   # card / cash / kakaopay / ...
    }


def _ensure_session(session_id: str | None = None):
    if session_id and session_id in SESSIONS and not _expired(SESS_META.get(session_id, 0)):
        ctx = SESSIONS[session_id]
    else:
        session_id = session_id or uuid.uuid4().hex
        ctx = _new_session_ctx()
        SESSIONS[session_id] = ctx
    SESS_META[session_id] = _now()
    return session_id, ctx


def _ctx_snapshot(ctx: Dict[str, Any]) -> dict:
    """프론트/백엔드에 내려줄 현재 상태 요약."""
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


def _reprompt_if_empty(text: str | None) -> str | None:
    """완전 공백일 때만 재질문. '네', '응' 같은 한 글자는 허용."""
    if text is None or not text.strip():
        return "죄송해요, 잘 못 들었어요. 다시 한 번 말씀해 주세요."
    return None


def _maybe_close_if_too_long(sid: str, ctx: Dict[str, Any]):
    """턴 수가 많아지면 세션 정리."""
    ctx["turns"] = ctx.get("turns", 0) + 1
    if ctx["turns"] > MAX_TURNS:
        resp = "대화가 길어져서 새로 시작할게요. 처음부터 다시 진행합니다."
        tts = synthesize(resp, out_path=f"response_{sid}.mp3")
        SESSIONS.pop(sid, None)
        SESS_META.pop(sid, None)
        return {
            "response_text": resp,
            "tts_path": tts,
            "tts_url": _make_tts_url(tts),
            "context": None,
            "backend_payload": None,
            "target_element_id": None,
        }
    return None


# ───────────────────────────────────────────────
# 파싱 함수들 (dine_type, category, menu, temp, size, options, payment)
# ───────────────────────────────────────────────
def _parse_dine_type(text: str) -> str | None:
    t = text.replace(" ", "")
    if "포장" in t or "들고갈" in t or "가져갈" in t:
        return "takeout"
    if "매장" in t or "먹고갈" in t or "여기서" in t:
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


def _menu_choices_for_category(cat: str):
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


def _parse_menu_item(category: str | None, text: str):
    if not category:
        return None
    t = text.replace(" ", "").lower()

    # 정확 매칭
    for mid, name in _menu_choices_for_category(category):
        key = name.replace(" ", "").lower()
        if key in t:
            return mid, name

    # 별칭
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
        if "치즈" in t:
            return "DESSERT_CHEESECAKE", "치즈케이크"
        if "티라" in t:
            return "DESSERT_TIRAMISU", "티라미수"
        if "브라우" in t:
            return "DESSERT_BROWNIE", "초코 브라우니"
        if "크루아" in t or "크로와상" in t:
            return "DESSERT_CROISSANT", "크루아상"
    return None


def _parse_temp(text: str) -> str | None:
    t = text.replace(" ", "")
    if "아이스" in t or "차갑" in t:
        return "ice"
    if "뜨겁" in t or "뜨거" in t or "따뜻" in t or "핫" in t:
        return "hot"
    return None


def _parse_size(text: str) -> str | None:
    t = text.replace(" ", "").lower()
    if "톨" in t:
        return "tall"
    if "그란데" in t:
        return "grande"
    if "벤티" in t:
        return "venti"
    if "스몰" in t:
        return "small"
    if "미디엄" in t:
        return "medium"
    if "라지" in t:
        return "large"
    return None


def _parse_options(category: str, text: str, options: dict):
    t = text.replace(" ", "").lower()

    if category == "coffee":
        # 디카페인
        if "디카" in t or "디카페인" in t:
            options["decaf"] = True
        # 시럽
        if "시럽" in t:
            options["syrup"] = True
        # 샷 추가
        if "샷" in t:
            if "두" in t or "2" in t:
                options["extra_shot"] = 2
            elif "세" in t or "3" in t:
                options["extra_shot"] = 3
            else:
                options["extra_shot"] = 1

    elif category == "ade":
        if "연하게" in t or "적게" in t:
            options["sweetness"] = "low"
        elif "보통" in t or "기본" in t:
            options["sweetness"] = "normal"
        elif "달게" in t or "많이" in t or "달달" in t:
            options["sweetness"] = "high"

    return options


def _parse_payment(text: str) -> str | None:
    t = text.replace(" ", "").lower()
    if "카드" in t:
        return "card"
    if "현금" in t:
        return "cash"
    if "카카오페이" in t:
        return "kakaopay"
    if "페이" in t:
        return "pay"
    return None


def _yes_no(text: str) -> str | None:
    t = text.replace(" ", "")
    if t in ("네", "응", "예", "맞아", "맞아요", "그래", "좋아요"):
        return "yes"
    if t in ("아니", "아니요", "싫어", "싫어요", "다시"):
        return "no"
    return None


# ───────────────────────────────────────────────
# backend_payload 생성
# ───────────────────────────────────────────────
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


# ───────────────────────────────────────────────
# UI 도움말용 LLM 프롬프트 & 함수 (target_element_id)
# ───────────────────────────────────────────────
UI_SYSTEM_PROMPT = """
너는 노인 친화 카페 키오스크의 음성 안내 도우미야.

- 사용자는 한국어로 버튼이나 영역이 "어디 있는지"를 물어본다.
- 너의 역할은:
  1) 사용자가 찾고 있는 UI 요소가 무엇인지 판단해서,
  2) 아래 목록 중 알맞은 target_element_id를 고르고,
  3) 그에 맞는 안내 문장을 answer_text에 넣어
  4) JSON으로만 반환하는 것이다.

가능한 target_element_id 목록:
메뉴 리스트 화면
- menu_home_button
- menu_pay_button
- menu_cart_area

온도 선택 화면
- temp_prev_button
- temp_next_button

사이즈 선택 화면
- size_prev_button
- size_next_button

옵션 선택 화면
- option_prev_button
- option_next_button

결제 리스트 화면 (주문 요약 모달)
- payment_prev_button
- payment_pay_button

QR 생성하기 팝업 (전화번호 입력)
- qr_cancel_button
- qr_send_button

규칙:
- target_element_id는 위 목록 중 하나만 사용해야 한다.
- 모르면 target_element_id에는 null을 넣고,
  answer_text는 "어느 버튼을 찾으시는지 다시 한번 말씀해 주세요."라고 해라.
- answer_text는 노인이 이해하기 쉽게, 존댓말로, 1~2문장으로 안내해라.
- 반드시 아래 JSON 형식으로만 출력해라. 다른 텍스트는 절대 쓰지 마라.

JSON 형식:
{
  "target_element_id": string | null,
  "answer_text": string
}
""".strip()

UI_FEW_SHOTS = """
예시 1)
사용자: 결제 버튼 어딨어?
응답:
{
  "target_element_id": "menu_pay_button",
  "answer_text": "메뉴 선택을 다 하셨으면, 화면 오른쪽 아래 파란색 ‘결제하기’ 버튼을 눌러 주세요."
}

예시 2)
사용자: 장바구니는 어디 있어?
응답:
{
  "target_element_id": "menu_cart_area",
  "answer_text": "화면 아래쪽 가운데에 있는 ‘장바구니’ 영역에서 주문하신 메뉴를 보실 수 있습니다."
}

예시 3)
사용자: 처음으로 돌아가는 거 어디야?
응답:
{
  "target_element_id": "menu_home_button",
  "answer_text": "화면 오른쪽 상단에 있는 동그란 ‘홈’ 버튼을 눌러 주세요."
}
""".strip()


def looks_like_ui_help(text: str) -> bool:
    """
    화면에서 버튼/영역 위치를 묻는 발화인지 간단 키워드로 감지.
    """
    t = text.replace(" ", "")
    keywords = [
        "버튼", "어디", "어딨어", "뒤로", "이전", "다음", "홈",
        "장바구니", "결제", "처음으로", "취소", "전송", "qr", "큐알"
    ]
    return any(k in t for k in keywords)


def classify_ui_target(user_text: str) -> dict:
    """
    OpenAI에 UI용 프롬프트로 물어보고
    { "target_element_id": ..., "answer_text": ... } 형태로 반환.
    """
    completion = gpt_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": UI_SYSTEM_PROMPT},
            {"role": "user", "content": UI_FEW_SHOTS},
            {"role": "user", "content": f"사용자: {user_text}\n응답:"},
        ],
        temperature=0.1,
        max_tokens=150,
    )

    raw = completion.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "target_element_id": None,
            "answer_text": "어느 버튼을 찾으시는지 다시 한번 말씀해 주세요."
        }

    # 방어적 필드 정리
    if "target_element_id" not in data:
        data["target_element_id"] = None
    if "answer_text" not in data:
        data["answer_text"] = "어느 버튼을 찾으시는지 다시 한번 말씀해 주세요."

    return data


# ───────────────────────────────────────────────
# OpenAI helper mode (대화형 자유 질문 답변)
# ───────────────────────────────────────────────
def looks_like_general_question(text: str) -> bool:
    """
    사용자가 메뉴/단계 외 일반 질문을 하는 상황 감지.
    예: '현금 돼?', '현금으로도 결제 돼?'
    (UI 위치 질문은 looks_like_ui_help가 먼저 처리함)
    """
    t = text.strip()

    # 결제 관련 질문
    if re.search(r"(현금|카드|결제)\s*(되|가능|돼)", t):
        return True

    # 안내 요청
    if "어떻게" in t or "방법" in t:
        return True

    # '메뉴 추천해줘', '뭐가 맛있어?' 등
    if re.search(r"(추천|맛있|뭐먹|뭐가)", t):
        return True

    # '?' 포함된 질문
    if t.endswith("?"):
        return True

    return False


def answer_general_question(text: str) -> str:
    """
    OpenAI API를 사용해 kiosk 안내 톤으로 대답 생성.
    """
    prompt = f"""
당신은 카페 키오스크 앞에서 손님을 도와주는 안내 도우미입니다.
손님이 한 질문: "{text}"

조건:
- 대답은 1~2문장, 짧고 명확하게.
- 너무 기술적인 설명은 피하고, 안내 말투로 설명.
- 존댓말 사용.
"""

    completion = gpt_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.6,
    )

    return completion.choices[0].message.content.strip()


# ───────────────────────────────────────────────
# 주문 요약 문장 생성
# ───────────────────────────────────────────────
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


# ───────────────────────────────────────────────
# 대화 흐름 함수 (주문 플로우 + 일반 질문)
# ───────────────────────────────────────────────
def _handle_turn(ctx: Dict[str, Any], user_text: str) -> str:
    text = (user_text or "").strip()
    step = ctx.get("step", "dine_type")
    category = ctx.get("category")

    # 일반 질문 감지 → OpenAI로 답변 (UI 위치 질문은 상위에서 이미 처리)
    if looks_like_general_question(text):
        return answer_general_question(text)

    # 1) 먹고가기 / 매장에서
    if step == "dine_type":
        dine = _parse_dine_type(text)
        if dine is None:
            return "포장해서 가져가시나요, 매장에서 드시나요?"
        ctx["dine_type"] = dine
        ctx["step"] = "menu_category"
        return "커피, 에이드, 차, 디저트 중 무엇을 드시겠어요?"

    # 2) 메뉴 카테고리
    if step == "menu_category":
        cat = _parse_category(text)
        if cat is None:
            return "커피, 에이드, 차, 디저트 중에서 선택해 주세요."
        ctx["category"] = cat
        ctx["step"] = "menu_item"
        names = " / ".join(n for _, n in _menu_choices_for_category(cat))
        return f"{names} 중에서 어떤 걸 드시겠어요?"

    # 3) 메뉴 선택
    if step == "menu_item":
        parsed = _parse_menu_item(category, text)
        if not parsed:
            names = " / ".join(n for _, n in _menu_choices_for_category(category))
            return f"다시 한 번 말씀해 주세요. 예: {names}"
        mid, name = parsed
        ctx["menu_id"] = mid
        ctx["menu_name"] = name

        if category in ("coffee", "tea"):
            ctx["step"] = "temp"
            return f"{name} 선택하셨어요. 따뜻하게 드실까요, 아이스로 드실까요?"
        if category == "ade":
            ctx["step"] = "size"
            return "사이즈는 톨, 그란데, 벤티 중에서 선택해 주세요."
        if category == "dessert":
            ctx["step"] = "confirm"
            return _order_summary_sentence(ctx)

    # 4) 온도 선택
    if step == "temp":
        temp = _parse_temp(text)
        if not temp:
            return "따뜻하게, 혹은 아이스로 말씀해 주세요."
        ctx["temp"] = temp
        ctx["step"] = "size"
        return "사이즈는 톨, 그란데, 벤티 중에서 선택해 주세요."

    # 5) 사이즈 선택
    if step == "size":
        size = _parse_size(text)
        if not size:
            return "사이즈를 다시 말씀해 주세요. 톨, 그란데, 벤티 중 하나를 말씀해 주세요."
        ctx["size"] = size

        if category in ("coffee", "ade"):
            ctx["step"] = "options"
            if category == "coffee":
                return "디카페인 여부, 샷 추가, 시럽 추가가 필요하시면 말씀해 주세요."
            else:
                return "당도는 연하게, 보통, 달게 중 어떤 걸로 할까요?"
        else:
            ctx["step"] = "confirm"
            return _order_summary_sentence(ctx)

    # 6) 옵션 선택
    if step == "options":
        options = ctx.get("options", {})
        ctx["options"] = _parse_options(category, text, options)
        ctx["step"] = "confirm"
        return _order_summary_sentence(ctx)

    # 7) 주문 확인
    if step == "confirm":
        yn = _yes_no(text)
        if yn == "yes":
            ctx["step"] = "payment"
            return "결제 수단은 카드, 현금, 카카오페이 중에서 선택해 주세요."
        if yn == "no":
            ctx.update(_new_session_ctx())
            ctx["step"] = "menu_category"
            return "알겠습니다. 처음부터 다시 할게요. 커피, 에이드, 차, 디저트 중에서 말씀해 주세요."
        return "주문이 맞으면 '네', 다시 선택하시려면 '아니요'라고 말씀해 주세요."

    # 8) 결제 수단
    if step == "payment":
        pay = _parse_payment(text)
        if not pay:
            return "결제 수단을 다시 말씀해 주세요. 카드, 현금, 카카오페이 등으로 말씀해 주세요."
        ctx["payment_method"] = pay
        ctx["step"] = "done"
        spoken = {
            "card": "카드",
            "cash": "현금",
            "kakaopay": "카카오페이",
            "pay": "간편결제",
        }.get(pay, "선택하신 결제 수단")
        return f"{spoken}로 결제 도와드릴게요. 주문이 완료되었습니다. 감사합니다."

    # 9) 주문 완료 후 새 주문
    if step == "done":
        ctx.update(_new_session_ctx())
        return "새 주문을 도와드릴게요. 포장해서 가져가시나요, 매장에서 드시나요?"

    # 비정상 상태 → 초기화
    ctx.update(_new_session_ctx())
    return "다시 처음부터 진행할게요. 포장해서 가져가시나요, 매장에서 드시나요?"


# ───────────────────────────────────────────────
# FastAPI 엔드포인트들
# ───────────────────────────────────────────────
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


@app.post("/session/start", response_model=StartOut)
def session_start():
    sid, ctx = _ensure_session()
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
        return {
            "stt_text": payload.text,
            "response_text": maybe,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": None,
        }

    # 턴 수 가드
    guard = _maybe_close_if_too_long(sid, ctx)
    if guard:
        return guard

    text = (payload.text or "").strip()

    # 1) 프론트에서 is_help=True를 보냈거나, UI 도움말로 보이는 발화면 → UI 모드
    if payload.is_help or looks_like_ui_help(text):
        ui_info = classify_ui_target(text)
        resp_text = ui_info.get(
            "answer_text",
            "어느 버튼을 찾으시는지 다시 한번 말씀해 주세요."
        )
        target_element_id = ui_info.get("target_element_id")

        tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
        SESS_META[sid] = _now()

        return {
            "stt_text": payload.text,
            "response_text": resp_text,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),           # 주문 상태는 유지
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": target_element_id,  # 프론트에서 하이라이트 용도로 사용
        }

    # 2) 그 외에는 기존 주문/일반 질문 흐름 사용
    resp_text = _handle_turn(ctx, payload.text)
    tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
    SESS_META[sid] = _now()

    return {
        "stt_text": payload.text,
        "response_text": resp_text,
        "tts_path": tts_path,
        "tts_url": _make_tts_url(tts_path) or None,
        "context": _ctx_snapshot(ctx),
        "backend_payload": _build_backend_payload(ctx),
        "target_element_id": None,
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
        return {
            "stt_text": user_text,
            "response_text": maybe,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": None,
        }

    # 턴 수 가드
    guard = _maybe_close_if_too_long(sid, ctx)
    if guard:
        return guard

    text = (user_text or "").strip()

    # 음성에서도 UI 도움말 발화면 같은 로직 적용
    if looks_like_ui_help(text):
        ui_info = classify_ui_target(text)
        resp_text = ui_info.get(
            "answer_text",
            "어느 버튼을 찾으시는지 다시 한번 말씀해 주세요."
        )
        target_element_id = ui_info.get("target_element_id")

        tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
        SESS_META[sid] = _now()

        return {
            "stt_text": user_text,
            "response_text": resp_text,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": target_element_id,
        }

    # 정상 주문/일반질문 처리
    resp_text = _handle_turn(ctx, user_text)
    tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
    SESS_META[sid] = _now()

    return {
        "stt_text": user_text,
        "response_text": resp_text,
        "tts_path": tts_path,
        "tts_url": _make_tts_url(tts_path) or None,
        "context": _ctx_snapshot(ctx),
        "backend_payload": _build_backend_payload(ctx),
        "target_element_id": None,
    }


@app.get("/session/state")
def session_state(session_id: str):
    if session_id not in SESSIONS or _expired(SESS_META.get(session_id, 0)):
        raise HTTPException(status_code=404, detail="세션 없음")
    ctx = SESSIONS[session_id]
    SESS_META[session_id] = _now()
    return _ctx_snapshot(ctx)


@app.get("/tts/{filename}")
def get_tts_file(filename: str):
    """생성된 TTS mp3를 내려주는 엔드포인트."""
    path = _tts_path_from_name(filename)
    return FileResponse(path, media_type="audio/mpeg", filename=filename)
