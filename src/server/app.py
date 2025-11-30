from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Iterable
import tempfile, os, uuid, time, re, asyncio
import io

from pydub import AudioSegment
from pydub.utils import which

from src.stt.whisper_client import transcribe_file
from src.tts.tts_client import synthesize
from src.pricing.price import load_configs

app = FastAPI(title="Voice Kiosk API", version="1.0.0")

# ── 세션/보안 가드 ──────────────────────────────────────────────────────────────
SESSIONS: Dict[str, Dict[str, Any]] = {}   # session_id -> context(dict)
SESS_META: Dict[str, float] = {}           # session_id -> last_active
SESSION_TTL = 600                          # 10분
MAX_TURNS = 20                             # 과도한 대화 방지
ACCEPTED_EXT = {".wav", ".mp3", ".m4a", ".3gp"}    # 업로드 허용 포맷

def _find_local_ffmpeg() -> str | None:
    tools_dir = os.path.abspath("tools")
    if not os.path.isdir(tools_dir):
        return None
    for entry in os.listdir(tools_dir):
        if not entry.lower().startswith("ffmpeg"):
            continue
        candidate_bin = os.path.join(tools_dir, entry, "bin")
        if not os.path.isdir(candidate_bin):
            continue
        exe_path = os.path.join(candidate_bin, "ffmpeg.exe")
        if os.path.isfile(exe_path):
            return exe_path
        unix_path = os.path.join(candidate_bin, "ffmpeg")
        if os.path.isfile(unix_path):
            return unix_path
    return None


def _ffprobe_from_ffmpeg(ffmpeg_path: str) -> str:
    base = os.path.dirname(ffmpeg_path)
    name = os.path.basename(ffmpeg_path)
    if name.lower().endswith(".exe"):
        return os.path.join(base, "ffprobe.exe")
    return os.path.join(base, "ffprobe")


def _resolve_ffmpeg_path() -> str | None:
    # 우선순위: 환경변수 → PATH → tools 폴더 → 일반적인 Windows 설치 경로
    candidates = [
        os.getenv("FFMPEG_BINARY"),
        which("ffmpeg"),
        which("ffmpeg.exe"),
        _find_local_ffmpeg(),
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


_FFMPEG_PATH = _resolve_ffmpeg_path()
if _FFMPEG_PATH:
    AudioSegment.converter = _FFMPEG_PATH
    AudioSegment.ffmpeg = _FFMPEG_PATH
    AudioSegment.ffprobe = _ffprobe_from_ffmpeg(_FFMPEG_PATH)

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


def _ensure_wav(path: str) -> tuple[str, list[str]]:
    """
    Whisper는 다양한 포맷을 지원하지만, 운영 편의를 위해 서버 내에서는
    항상 WAV로 변환된 파일 경로를 사용한다.
    """
    cleanup_targets = [path]
    if path.lower().endswith(".wav"):
        return path, cleanup_targets

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        wav_path = tmp_wav.name

    try:
        audio = AudioSegment.from_file(path)
        audio.export(wav_path, format="wav")
    except FileNotFoundError as exc:
        # 주로 ffmpeg 바이너리를 찾지 못했을 때 발생
        try:
            os.remove(wav_path)
        except OSError:
            pass
        err_msg = (
            "오디오 변환 실패: ffmpeg 실행 파일을 찾을 수 없습니다. "
            "시스템 PATH에 ffmpeg를 추가하거나 환경변수 FFMPEG_BINARY를 설정해 주세요."
        )
        raise HTTPException(status_code=500, detail=err_msg) from exc
    except Exception as exc:
        # 생성 실패 시 임시 WAV도 정리
        try:
            os.remove(wav_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=f"오디오 변환 실패: {exc}")

    cleanup_targets.append(wav_path)
    return wav_path, cleanup_targets


def _cleanup_temp_files(paths: Iterable[str]) -> None:
    """임시 파일 삭제 시 예외를 무시하고 진행."""
    for path in paths:
        if not path:
            continue
        try:
            os.remove(path)
        except OSError:
            pass


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
        # greeting -> dine_type -> menu_item -> temp/size -> options -> add_more -> review -> phone -> payment -> card -> done
        "step": "greeting",
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


def _parse_menu_item(category: str | None, text: str) -> tuple[str, str, str] | None:
    """사용자 발화에서 메뉴를 찾아 (category, menu_id, menu_name) 반환. category가 None이면 모든 카테고리에서 검색."""
    t = text.replace(" ", "").lower()
    categories_to_search = [category] if category else ["coffee", "ade", "tea", "dessert"]
    
    for cat in categories_to_search:
        # 정확한 메뉴명 매칭
        for mid, name in _menu_choices_for_category(cat):
            key = name.replace(" ", "").lower()
            if key in t:
                return cat, mid, name
        
        # 별칭 처리 (발음 변형 포함)
        if cat == "coffee":
            # 아메리카노: 아메리카노, 아메리까노, 아메레카노, 아메리코노, 아메르카노, 아메리카노우, 아메라고, 아메르가노, 아메니카노, 아메리카루, 아메리노
            if any(x in t for x in ["아메리카노", "아메리까노", "아메레카노", "아메리코노", "아메르카노", "아메리카노우", "아메라고", "아메르가노", "아메니카노", "아메리카루", "아메리노", "아메"]):
                return "coffee", "COFFEE_AMERICANO", "아메리카노"
            # 에스프레소: 에스프레소, 에스뿌레소, 에스쁘레소, 에스프래소, 에스프라쏘, 에스플레소, 에스프레수, 에스프레쏘오, 에스프로소, 에스뿌레쏘
            if any(x in t for x in ["에스프레소", "에스뿌레소", "에스쁘레소", "에스프래소", "에스프라쏘", "에스플레소", "에스프레수", "에스프레쏘오", "에스프로소", "에스뿌레쏘"]) or \
               (("에스프" in t or "애스프" in t or "에스뿌" in t or "에스쁘" in t) and any(x in t for x in ["레소", "라소", "래소", "래쏘", "레쏘", "레쏘오", "로소", "레수"])):
                return "coffee", "COFFEE_ESPRESSO", "에스프레소"
            # 카페 라떼: 라떼이, 라테이, 라떼요, 라테요, 라떼우, 라테우, 카페라떼, 카페라테, 카페라뗴, 카페라떼이
            if any(x in t for x in ["라떼이", "라테이", "라떼요", "라테요", "라떼우", "라테우", "카페라떼", "카페라테", "카페라뗴", "카페라떼이", "라떼", "라테"]):
                return "coffee", "COFFEE_LATTE", "카페 라떼"
            # 카푸치노: 카푸치노우, 카푸치노오, 카푸찌노, 카푸치노어, 카프치노, 카뿌치노
            if any(x in t for x in ["카푸치노우", "카푸치노오", "카푸찌노", "카푸치노어", "카프치노", "카뿌치노", "카푸치노", "카푸"]):
                return "coffee", "COFFEE_CAPPUCCINO", "카푸치노"
        if cat == "ade":
            # 레몬에이드: 레몬에이, 레몬에이두, 레몬에이더, 레몬애이드, 레몬네이드, 레몬네이, 레몽에이드, 레멍에이드
            if any(x in t for x in ["레몬에이", "레몬에이두", "레몬에이더", "레몬애이드", "레몬네이드", "레몬네이", "레몽에이드", "레멍에이드", "레몬", "레몽"]):
                return "ade", "ADE_LEMON", "레몬에이드"
            # 자몽에이드: 자몽에이, 자몽에이더, 자몽애이드, 자몽네이드, 자몽네이, 자몽에이두, 자몽에두, 자뭉에이드
            if any(x in t for x in ["자몽에이", "자몽에이더", "자몽애이드", "자몽네이드", "자몽네이", "자몽에이두", "자몽에두", "자뭉에이드", "자몽"]):
                return "ade", "ADE_GRAPEFRUIT", "자몽에이드"
            # 청포도 에이드: 청포도에이, 청포도에이더, 청포도네이드, 청포도네이, 청포도에이두, 청포도에두, 쳥포도 에이드, 청포도에듀
            if any(x in t for x in ["청포도에이", "청포도에이더", "청포도네이드", "청포도네이", "청포도에이두", "청포도에두", "쳥포도", "청포도에듀", "청포도"]):
                return "ade", "ADE_GREEN_GRAPE", "청포도 에이드"
            # 오렌지 에이드: 오렌지에이, 오렌지에이더, 오렌지네이드, 오렌지네이, 오렌지애이드, 오랜지 에이드, 오렌지에두, 오렌지두
            if any(x in t for x in ["오렌지에이", "오렌지에이더", "오렌지네이드", "오렌지네이", "오렌지애이드", "오랜지", "오렌지에두", "오렌지두", "오렌지"]):
                return "ade", "ADE_ORANGE", "오렌지 에이드"
        if cat == "tea":
            # 캐모마일: 카모마일, 카모마일티, 카모, 캐모마일티, 캐모마일트, 캐모말, 캐모마, 케모마일, 카모메일
            if any(x in t for x in ["카모마일", "카모마일티", "카모", "캐모마일티", "캐모마일트", "캐모말", "캐모마", "케모마일", "카모메일", "캐모마일", "캐모"]):
                return "tea", "TEA_CHAMOMILE", "캐모마일 티"
            # 얼그레이: 얼그레이이, 얼그레, 얼그레잉, 얼그레잇, 얼그레에, 얼그레어, 얼그레오, 얼그레히, 얼글레이, 얼끌레이
            if any(x in t for x in ["얼그레이이", "얼그레", "얼그레잉", "얼그레잇", "얼그레에", "얼그레어", "얼그레오", "얼그레히", "얼글레이", "얼끌레이", "얼그레이", "얼그"]):
                return "tea", "TEA_EARL_GREY", "얼그레이 티"
            # 유자차: 유자챠, 유자차이, 유자차우, 유자타, 유자자, 유자티
            if any(x in t for x in ["유자챠", "유자차이", "유자차우", "유자타", "유자자", "유자티", "유자"]):
                return "tea", "TEA_YUJA", "유자차"
            # 녹차: 녹챠, 녹차이, 녹차우, 녹차어, 눅차, 녹타, 록차
            if any(x in t for x in ["녹챠", "녹차이", "녹차우", "녹차어", "눅차", "녹타", "록차", "녹차"]):
                return "tea", "TEA_GREEN", "녹차"
        if cat == "dessert":
            # 치즈케이크: 치즈케키, 치즈케잌, 치즈케익, 치즈케잌크, 치즈케에크, 치케, 치즈케이, 치즈케에익, 지즈케이크, 치츠케이크
            if any(x in t for x in ["치즈케키", "치즈케잌", "치즈케익", "치즈케잌크", "치즈케에크", "치케", "치즈케이", "치즈케에익", "지즈케이크", "치츠케이크", "치즈케이크", "치즈케", "치즈"]):
                return "dessert", "DESSERT_CHEESECAKE", "치즈케이크"
            # 티라미수: 티라미슈, 티라미스, 티람이수, 티라미쑤우, 티라미소, 티라미쓰, 티라미슈우, 디라미수
            if any(x in t for x in ["티라미슈", "티라미스", "티람이수", "티라미쑤우", "티라미소", "티라미쓰", "티라미슈우", "디라미수", "티라미수", "티라미쑤", "티라"]):
                return "dessert", "DESSERT_TIRAMISU", "티라미수"
            # 브라우니: 브라운니, 브라오니, 브라우니이, 브라우니우, 브라우닝, 브라오니, 브라운이
            if any(x in t for x in ["브라운니", "브라오니", "브라우니이", "브라우니우", "브라우닝", "브라운이", "브라우니", "브라우"]):
                return "dessert", "DESSERT_BROWNIE", "초코 브라우니"
            # 크루아상: 크루와상, 크로와상, 크로아상, 크루아쌍, 크루아쌍그, 크루아송, 크루아샹, 크로와쌍
            if any(x in t for x in ["크루와상", "크로와상", "크로아상", "크루아쌍", "크루아쌍그", "크루아송", "크루아샹", "크로와쌍", "크루아상", "크루아"]):
                return "dessert", "DESSERT_CROISSANT", "크루아상"
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


def _order_confirmation_sentence(ctx: Dict[str, Any]) -> str:
    """
    옵션 선택 완료 후 확인 메시지 생성
    형식: "주문하신 음료가 [메뉴명] [온도]/[사이즈]/[옵션]가 맞으신가요?"
    """
    category = ctx.get("category")
    menu_name = ctx.get("menu_name") or {
        "coffee": "커피",
        "ade": "에이드",
        "tea": "차",
        "dessert": "디저트",
    }.get(category, "메뉴")

    temp = ctx.get("temp")
    size = ctx.get("size")
    options = ctx.get("options", {}) or {}

    # 온도 문자열
    temp_str = ""
    if temp == "ice":
        temp_str = "아이스"
    elif temp == "hot":
        temp_str = "따뜻하게"

    # 사이즈 문자열
    size_str = {
        "tall": "톨",
        "grande": "그란데",
        "venti": "벤티",
        "small": "스몰",
        "medium": "미디엄",
        "large": "라지",
    }.get(size, "")

    # 옵션 문자열
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

    # 슬래시로 구분된 정보 조합
    parts: list[str] = [menu_name]
    if temp_str:
        parts.append(temp_str)
    if size_str:
        parts.append(size_str)
    if opt_parts:
        parts.extend(opt_parts)

    order_info = "/".join(parts)
    return f"주문하신 음료가 {order_info}가 맞으신가요?"


def _cart_added_sentence(ctx: Dict[str, Any]) -> str:
    """
    장바구니 담김 메시지 생성
    형식: "에스프레소, 차갑게/벤티/시럽추가가 장바구니에 담겼습니다..."
    """
    category = ctx.get("category")
    menu_name = ctx.get("menu_name") or {
        "coffee": "커피",
        "ade": "에이드",
        "tea": "차",
        "dessert": "디저트",
    }.get(category, "메뉴")

    temp = ctx.get("temp")
    size = ctx.get("size")
    options = ctx.get("options", {}) or {}

    # 온도 문자열
    temp_str = ""
    if temp == "ice":
        temp_str = "차갑게"
    elif temp == "hot":
        temp_str = "따뜻하게"

    # 사이즈 문자열
    size_str = {
        "tall": "톨",
        "grande": "그란데",
        "venti": "벤티",
        "small": "스몰",
        "medium": "미디엄",
        "large": "라지",
    }.get(size, "")

    # 옵션 문자열
    opt_parts: list[str] = []
    if category == "coffee":
        if options.get("decaf"):
            opt_parts.append("디카페인")
        if options.get("extra_shot", 0) > 0:
            opt_parts.append(f"샷 {options['extra_shot']}번 추가")
        if options.get("syrup"):
            opt_parts.append("시럽추가")
    elif category == "ade":
        sweetness = options.get("sweetness")
        if sweetness == "low":
            opt_parts.append("당도 낮게")
        elif sweetness == "normal":
            opt_parts.append("당도 보통")
        elif sweetness == "high":
            opt_parts.append("당도 높게")

    # 메뉴명과 옵션 정보 조합 (쉼표로 메뉴명 구분, 슬래시로 옵션 구분)
    parts: list[str] = []
    if temp_str:
        parts.append(temp_str)
    if size_str:
        parts.append(size_str)
    if opt_parts:
        parts.extend(opt_parts)

    if parts:
        order_info = f"{menu_name}, {"/".join(parts)}"
    else:
        order_info = menu_name

    return f"{order_info}가 장바구니에 담겼습니다. 이어서 주문을 진행하시거나 결제하기 버튼을 눌러주세요."


# ── 메인 대화 흐름 로직 ──────────────────────────────────────────────────────
def _handle_turn(ctx: Dict[str, Any], user_text: str) -> str:
    """
    한 턴의 입력(user_text)에 대해,
    ctx 상태를 업데이트하고, 사용자에게 들려줄 response_text를 반환.
    """
    text = (user_text or "").strip()
    step = ctx.get("step", "greeting")
    category = ctx.get("category")

    # 0) 인사 단계
    if step == "greeting":
        # "주문" 키워드 확인
        if "주문" in text or "시작" in text or "시작할게" in text:
            ctx["step"] = "dine_type"
            return "포장해서 가져가시나요, 매장에서 드시나요?"
        # 주문 버튼을 누르지 않았으면 인사 메시지 반환
        return "안녕하세요. AI음성 키오스크 말로입니다. 주문을 도와드릴게요."

    # 1) 먹고가기 / 들고가기
    if step == "dine_type":
        dine = _parse_dine_type(text)
        if dine is None:
            ctx["step"] = "dine_type"
            return "포장해서 가져가시나요, 매장에서 드시나요?"
        ctx["dine_type"] = dine
        
        # 선택한 옵션을 한국어로 변환
        dine_name = "들고가기" if dine == "takeout" else "먹고가기"
        
        ctx["step"] = "menu_item"
        return f"{dine_name}를 선택하셨습니다. 원하시는 메뉴를 말씀해주세요."

    # 2) 세부 메뉴 선택 (아메리카노, 레몬에이드, 치즈케이크 등)
    if step == "menu_item":
        # 결제하기 버튼 클릭 체크
        t = text.replace(" ", "").lower()
        is_payment_intent = any(x in t for x in ["결제하기", "결제", "결제할게요", "결제하겠어요", "결제하겠습니다"])
        
        if is_payment_intent:
            # 주문 내역이 있는지 확인
            if ctx.get("menu_name") and ctx.get("category"):
                # 주문 내역이 있으면 확인 단계로
                ctx["step"] = "confirm"
                return "주문내역을 확인하고 결제를 진행해주세요."
            else:
                # 주문 내역이 없으면 메뉴 선택 요청
                return "주문하실 메뉴를 먼저 선택해 주세요."
        
        parsed = _parse_menu_item(category, text)
        if not parsed:
            return "죄송해요, 잘 못 들었어요. 다시 한 번 메뉴를 말씀해 주세요."
        parsed_category, menu_id, menu_name = parsed
        ctx["category"] = parsed_category
        ctx["menu_id"] = menu_id
        ctx["menu_name"] = menu_name
        ctx["temp"] = None
        ctx["size"] = None
        ctx["options"] = {
            "extra_shot": 0,
            "syrup": False,
            "decaf": None,
            "sweetness": None,
        }

        # 카테고리별로 다음 단계 분기
        category = parsed_category
        if category in ("coffee", "tea"):
            ctx["step"] = "temp"
            return f"{menu_name}를 선택하셨어요. 따뜻하게 드실까요, 차갑게 드실까요?"
        if category == "ade":
            ctx["step"] = "size"
            return f"{menu_name}를 선택하셨어요. 사이즈는 작은 사이즈, 중간 사이즈, 큰 사이즈 중에서 선택해 주세요."
        if category == "dessert":
            ctx["step"] = "confirm"
            return _order_summary_sentence(ctx)

    # 4) 온도 선택 (커피/차)
    if step == "temp":
        # 이전 버튼 클릭 체크
        t = text.replace(" ", "").lower()
        is_back = any(x in t for x in ["이전", "뒤로", "취소", "돌아가", "back", "prev"])
        
        if is_back:
            ctx["step"] = "menu_item"
            return "주문을 다시 진행해주세요."
        
        temp = _parse_temp(text)
        if temp is None:
            return "따뜻하게 드실지, 차갑게 드실지 말씀해 주세요. 예: '아이스로 주세요'."
        ctx["temp"] = temp
        ctx["step"] = "size"
        how = "아이스" if temp == "ice" else "뜨겁게"
        return f"{how}로 준비할게요. 사이즈는 작은 사이즈, 중간 사이즈, 큰 사이즈 중에서 선택해 주세요."

    # 5) 사이즈 선택
    if step == "size":
        # 이전 버튼 클릭 체크
        t = text.replace(" ", "").lower()
        is_back = any(x in t for x in ["이전", "뒤로", "취소", "돌아가", "back", "prev"])
        
        if is_back:
            # 온도 선택이 필요한 카테고리인 경우
            if category in ("coffee", "tea"):
                ctx["step"] = "temp"
                return "온도를 다시 선택해주세요."
            # 에이드는 온도 선택 없이 사이즈만 선택하므로 메뉴 선택으로
            else:
                ctx["step"] = "menu_item"
                return "주문을 다시 진행해주세요."
        
        size = _parse_size(text)
        if size is None:
            return "사이즈를 다시 말씀해 주세요. 작은 사이즈, 중간 사이즈, 큰 사이즈 중 하나를 선택해 주세요."
        ctx["size"] = size

        # 사이즈를 한국어로 변환
        size_map = {
            "tall": "톨",
            "grande": "그란데",
            "venti": "벤티",
            "small": "작은사이즈",
            "medium": "중간사이즈",
            "large": "큰사이즈",
        }
        size_name = size_map.get(size, "사이즈")

        if category == "coffee":
            ctx["step"] = "options"
            return f"{size_name}를 선택하였습니다. 옵션을 선택해주세요."
        if category == "ade":
            ctx["step"] = "options"
            return f"{size_name}를 선택하였습니다. 옵션을 선택해주세요."
        if category == "tea":
            ctx["step"] = "confirm"
            return _order_summary_sentence(ctx)
        if category == "dessert":
            ctx["step"] = "confirm"
            return _order_summary_sentence(ctx)

    # 6) 옵션 선택 (커피/에이드)
    if step == "options":
        # 이전 버튼 클릭 체크
        t = text.replace(" ", "").lower()
        is_back = any(x in t for x in ["이전", "뒤로", "취소", "돌아가", "back", "prev"])
        
        if is_back:
            ctx["step"] = "size"
            return "사이즈를 다시 선택해주세요."
        
        options = ctx.get("options", {})
        ctx["options"] = _parse_options(category, text, options)
        # 옵션 선택 후 메뉴 정보는 유지하고 메뉴판으로 돌아감
        ctx["step"] = "menu_item"
        return _cart_added_sentence(ctx)

    # 7) 주문 내역 확인
    if step == "confirm":
        # 이전 버튼 클릭 체크
        t = text.replace(" ", "").lower()
        is_back = any(x in t for x in ["이전", "뒤로", "취소", "돌아가", "back", "prev"])
        
        if is_back:
            ctx["step"] = "menu_item"
            return "주문을 계속 진행해주세요."
        
        # 결제하기 버튼 클릭 또는 결제 관련 키워드 체크
        is_payment_intent = any(x in t for x in ["결제하기", "결제", "결제할게요", "결제하겠어요", "결제하겠습니다"])
        
        print(f"받은 텍스트: {text}")
        print(f"전처리 후: {t}")
        print(f"is_payment_intent: {is_payment_intent}")
        
        yn = _yes_no(text)
        if yn == "yes" or is_payment_intent:
            ctx["step"] = "payment"
            return "결제 수단을 선택해 주세요. 카드결제, 간편결제, 쿠폰 사용 등으로 말씀해 주세요."
        if yn == "no":
            # 메뉴부터 다시
            ctx["category"] = None
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
            return "알겠습니다. 다시 원하시는 메뉴를 말씀해 주세요."
        return "주문이 맞으면 '네', 다시 선택하시려면 '아니요'라고 말씀해 주세요."

    # 8) 결제 수단 선택
    if step == "payment":
        # 이전 버튼 클릭 체크
        t = text.replace(" ", "").lower()
        is_back = any(x in t for x in ["이전", "뒤로", "취소", "돌아가", "back", "prev"])
        
        if is_back:
            ctx["step"] = "menu_item"
            return "주문을 계속 진행해주세요."
        
        pay = _parse_payment(text)
        if pay is None:
            return "결제 수단을 다시 말씀해 주세요. 카드결제, 간편결제, 쿠폰 사용 등으로 말씀해 주세요."
        ctx["payment_method"] = pay
        
        # 카드 결제인 경우 card 단계로
        if pay == "card":
            ctx["step"] = "card"
            return "카드를 삽입해주세요."
        
        # 그 외 결제 수단은 바로 완료
        ctx["step"] = "done"
        spoken_pay = {
            "pay": "간편결제",
            "kakaopay": "카카오페이",
            "samsungpay": "삼성페이",
            "coupon": "쿠폰",
        }.get(pay, "선택하신 결제 수단")
        return f"{spoken_pay}로 결제 도와드릴게요. 주문이 완료되었습니다. 감사합니다."

    # 9) 카드 삽입 및 결제 완료
    if step == "card":
        # 카드 삽입 완료 확인 (예: "카드 넣었어요", "완료", "결제됐어요" 등)
        t = text.replace(" ", "").lower()
        is_complete = any(x in t for x in ["완료", "됐", "넣었", "삽입", "결제", "다됐"])
        
        if is_complete:
            ctx["step"] = "done"
            return "결제가 완료되었습니다. 카드를 제거해주세요."
        return "카드를 삽입해주세요."

    # 10) 주문 완료 후 새 주문
    if step == "done":
        ctx.clear()
        ctx.update(_new_session_ctx())
        return "안녕하세요. AI음성 키오스크 말로입니다. 주문을 도와드릴게요."

    # 안전장치: 알 수 없는 상태면 처음으로
    ctx.clear()
    ctx.update(_new_session_ctx())
    return "안녕하세요. AI음성 키오스크 말로입니다. 주문을 도와드릴게요."


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
    # step을 명시적으로 "greeting"으로 설정
    ctx["step"] = "greeting"
    # _handle_turn을 호출하여 greeting 단계 응답 받기
    resp_text = _handle_turn(ctx, "")
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

    cleanup_paths: list[str] = [tmp_path]
    try:
        wav_path, cleanup_paths = _ensure_wav(tmp_path)
    except HTTPException:
        # 변환 실패 시에도 원본 파일 제거 필요
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    try:
        user_text = transcribe_file(wav_path, language="ko")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"STT 실패: {e}")
    finally:
        _cleanup_temp_files(cleanup_paths)

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


# ── WebSocket 실시간 음성 스트리밍 ────────────────────────────────────────────────
@app.websocket("/session/voice/stream")
async def session_voice_stream(websocket: WebSocket, session_id: str):
    """
    실시간 음성 스트리밍을 위한 WebSocket 엔드포인트.
    프론트엔드에서 음성 데이터를 실시간으로 전송하면 STT 처리 후 응답을 반환.
    """
    await websocket.accept()
    sid, ctx = _ensure_session(session_id)
    
    # 음성 데이터 버퍼
    audio_buffer = io.BytesIO()
    last_process_time = time.time()
    last_data_time = time.time()
    PROCESS_INTERVAL = 2.0  # 2초마다 STT 처리
    SILENCE_TIMEOUT = 3.0  # 3초간 데이터가 없으면 처리
    MIN_BUFFER_SIZE = 1024  # 최소 버퍼 크기 (더 작게 설정)
    
    print(f"[WebSocket] 세션 {sid} 스트리밍 시작")
    
    try:
        while True:
            # 음성 데이터 수신
            try:
                # 타임아웃을 더 길게 설정 (0.5초)
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.5)
                audio_buffer.write(data)
                last_data_time = time.time()
                current_time = time.time()
                buffer_size = audio_buffer.tell()
                
                print(f"[WebSocket] 데이터 수신: {len(data)} bytes, 총 버퍼: {buffer_size} bytes")
                
                # 일정 시간이 지났거나 버퍼가 충분히 쌓였으면 STT 처리
                time_elapsed = current_time - last_process_time
                silence_elapsed = current_time - last_data_time
                
                should_process = (
                    (time_elapsed >= PROCESS_INTERVAL and buffer_size >= MIN_BUFFER_SIZE) or
                    (buffer_size >= MIN_BUFFER_SIZE * 4) or  # 버퍼가 너무 크면 강제 처리
                    (silence_elapsed >= SILENCE_TIMEOUT and buffer_size >= MIN_BUFFER_SIZE)  # 침묵 후 처리
                )
                
                if should_process:
                    print(f"[WebSocket] STT 처리 시작: 버퍼 크기={buffer_size}, 경과 시간={time_elapsed:.2f}초")
                    
                    # 버퍼 내용을 임시 파일로 저장
                    audio_buffer.seek(0)
                    buffer_data = audio_buffer.read()
                    
                    if len(buffer_data) > 0:
                        # 임시 파일 생성 (원본 포맷 유지)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".3gp") as tmp:
                            tmp.write(buffer_data)
                            tmp_path = tmp.name
                        
                        cleanup_paths = [tmp_path]
                        try:
                            # WAV 형식으로 변환 (필요시)
                            wav_path, cleanup_paths = _ensure_wav(tmp_path)
                            
                            # STT 처리
                            try:
                                user_text = transcribe_file(wav_path, language="ko")
                                print(f"[WebSocket] STT 결과: {user_text}")
                                
                                # 무음 처리
                                maybe = _reprompt_if_empty(user_text)
                                if maybe:
                                    await websocket.send_json({
                                        "type": "reprompt",
                                        "response_text": maybe,
                                        "stt_text": user_text,
                                    })
                                else:
                                    # 턴 수 가드
                                    guard = _maybe_close_if_too_long(sid, ctx)
                                    if guard:
                                        await websocket.send_json({
                                            "type": "session_closed",
                                            "response_text": guard["response_text"],
                                        })
                                        break
                                    
                                    # 대화 처리
                                    resp_text = _handle_turn(ctx, user_text)
                                    tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
                                    SESS_META[sid] = _now()
                                    backend_payload = _build_backend_payload(ctx)
                                    
                                    await websocket.send_json({
                                        "type": "response",
                                        "stt_text": user_text,
                                        "response_text": resp_text,
                                        "tts_path": tts_path,
                                        "tts_url": _make_tts_url(tts_path) or None,
                                        "context": _ctx_snapshot(ctx),
                                        "backend_payload": backend_payload,
                                    })
                                
                            except Exception as e:
                                print(f"[WebSocket] STT 처리 오류: {e}")
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"STT 처리 실패: {str(e)}",
                                })
                            finally:
                                _cleanup_temp_files(cleanup_paths)
                        
                        except Exception as e:
                            print(f"[WebSocket] 오디오 변환 오류: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": f"오디오 변환 실패: {str(e)}",
                            })
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
                    
                    # 버퍼 초기화
                    audio_buffer = io.BytesIO()
                    last_process_time = current_time
                    print(f"[WebSocket] 버퍼 초기화 완료")
                
            except asyncio.TimeoutError:
                # 타임아웃은 정상 (데이터가 없을 때)
                # 하지만 너무 오래 데이터가 없으면 버퍼 처리
                current_time = time.time()
                buffer_size = audio_buffer.tell()
                silence_elapsed = current_time - last_data_time
                
                if buffer_size >= MIN_BUFFER_SIZE and silence_elapsed >= SILENCE_TIMEOUT:
                    print(f"[WebSocket] 침묵 타임아웃으로 처리: 버퍼={buffer_size}, 침묵={silence_elapsed:.2f}초")
                    # 위의 처리 로직과 동일하게 처리
                    audio_buffer.seek(0)
                    buffer_data = audio_buffer.read()
                    
                    if len(buffer_data) > 0:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".3gp") as tmp:
                            tmp.write(buffer_data)
                            tmp_path = tmp.name
                        
                        cleanup_paths = [tmp_path]
                        try:
                            wav_path, cleanup_paths = _ensure_wav(tmp_path)
                            try:
                                user_text = transcribe_file(wav_path, language="ko")
                                print(f"[WebSocket] STT 결과: {user_text}")
                                
                                maybe = _reprompt_if_empty(user_text)
                                if maybe:
                                    await websocket.send_json({
                                        "type": "reprompt",
                                        "response_text": maybe,
                                        "stt_text": user_text,
                                    })
                                else:
                                    guard = _maybe_close_if_too_long(sid, ctx)
                                    if guard:
                                        await websocket.send_json({
                                            "type": "session_closed",
                                            "response_text": guard["response_text"],
                                        })
                                        break
                                    
                                    resp_text = _handle_turn(ctx, user_text)
                                    tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
                                    SESS_META[sid] = _now()
                                    backend_payload = _build_backend_payload(ctx)
                                    
                                    await websocket.send_json({
                                        "type": "response",
                                        "stt_text": user_text,
                                        "response_text": resp_text,
                                        "tts_path": tts_path,
                                        "tts_url": _make_tts_url(tts_path) or None,
                                        "context": _ctx_snapshot(ctx),
                                        "backend_payload": backend_payload,
                                    })
                            except Exception as e:
                                print(f"[WebSocket] STT 처리 오류: {e}")
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"STT 처리 실패: {str(e)}",
                                })
                            finally:
                                _cleanup_temp_files(cleanup_paths)
                        except Exception as e:
                            print(f"[WebSocket] 오디오 변환 오류: {e}")
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
                    
                    audio_buffer = io.BytesIO()
                    last_process_time = current_time
                    last_data_time = current_time
                
                continue
            except WebSocketDisconnect:
                print(f"[WebSocket] 클라이언트 연결 종료")
                break
            except Exception as e:
                print(f"[WebSocket] 처리 중 오류: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"처리 중 오류: {str(e)}",
                })
                break
    
    except Exception as e:
        print(f"[WebSocket] 연결 오류: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"연결 오류: {str(e)}",
            })
        except:
            pass
    finally:
        print(f"[WebSocket] 세션 {sid} 스트리밍 종료")
        try:
            await websocket.close()
        except:
            pass
