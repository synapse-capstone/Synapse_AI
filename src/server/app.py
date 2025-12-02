from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Iterable
import tempfile, os, uuid, time, re, json

from pydub import AudioSegment
from pydub.utils import which
from openai import OpenAI

from src.stt.whisper_client import transcribe_file, _make_client as make_whisper_client
from src.tts.tts_client import synthesize
from src.pricing.price import load_configs

app = FastAPI(title="Voice Kiosk API", version="1.0.0")

# ── 세션/보안 가드 ──────────────────────────────────────────────────────────────
SESSIONS: Dict[str, Dict[str, Any]] = {}   # session_id -> context(dict)
SESS_META: Dict[str, float] = {}           # session_id -> last_active
SESSION_TTL = 600                          # 10분
MAX_TURNS = 20                             # 과도한 대화 방지
ACCEPTED_EXT = {".wav", ".mp3", ".m4a", ".3gp"}    # 업로드 허용 포맷

# OpenAI 클라이언트 (환경변수 OPENAI_API_KEY 사용)
gpt_client = OpenAI()

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



def _ensure_wav(path: str) -> tuple[str, list[str]]:
    """
    Whisper는 다양한 포맷을 지원하지만, 운영 편의를 위해 서버 내에서는
    항상 WAV로 변환된 파일 경로를 사용한다.
    3gp 파일은 명시적으로 포맷을 지정하여 변환한다.
    """
    cleanup_targets = [path]
    if path.lower().endswith(".wav"):
        return path, cleanup_targets

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        wav_path = tmp_wav.name

    try:
        # 파일 확장자에 따라 포맷 명시
        file_ext = os.path.splitext(path)[1].lower()
        
        # 3gp 파일은 명시적으로 포맷 지정
        if file_ext == ".3gp":
            # 3gp는 AMR 또는 AAC 코덱을 사용할 수 있으므로 포맷을 명시
            audio = AudioSegment.from_file(path, format="3gp")
        elif file_ext == ".m4a":
            audio = AudioSegment.from_file(path, format="m4a")
        elif file_ext == ".mp3":
            audio = AudioSegment.from_file(path, format="mp3")
        else:
            # 기타 포맷은 자동 감지
            audio = AudioSegment.from_file(path)
        
        # WAV로 변환 (16kHz, mono로 정규화하여 Whisper에 최적화)
        audio = audio.set_frame_rate(16000).set_channels(1)
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
        # 더 자세한 오류 메시지
        error_detail = str(exc)
        if "Invalid data" in error_detail or "Invalid" in error_detail:
            error_detail = f"오디오 파일이 손상되었거나 지원되지 않는 형식입니다: {error_detail}"
        raise HTTPException(status_code=400, detail=f"오디오 변환 실패 ({file_ext}): {error_detail}")

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
        # 대화 단계:
        # greeting -> dine_type -> menu_item -> temp/size -> options -> add_more -> review -> phone -> payment -> card -> done
        "step": "greeting",
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
    snapshot = {
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
    # 최근 응답 정보가 있으면 포함
    if "last_response" in ctx:
        snapshot["last_response"] = ctx.get("last_response")
    return snapshot


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
            ("DESSERT_MACARON", "마카롱"),
        ]
    return []


def _parse_menu_item(category: str | None, text: str) -> tuple[str, str, str] | None:
    """
    사용자 발화에서 메뉴를 찾아 (category, menu_id, menu_name) 반환.
    category가 지정되어 있어도, 해당 카테고리에서 찾지 못하면 전체 카테고리에서 검색.
    """
    # 공백 제거 및 소문자 변환 (한글은 소문자 변환이 없지만 일관성을 위해)
    t = text.replace(" ", "").replace(".", "").replace(",", "").lower()
    
    print(f"[메뉴 파싱] 입력 텍스트: '{text}' (정규화: '{t}'), 카테고리: {category or '전체'}")
    
    # 1단계: category가 지정되어 있으면 먼저 해당 카테고리에서 검색
    if category:
        categories_to_search = [category]
        for cat in categories_to_search:
            # 정확한 메뉴명 매칭 (공백 제거 후 포함 여부 확인)
            for mid, name in _menu_choices_for_category(cat):
                # 메뉴명에서 공백과 "티" 제거 (예: "캐모마일 티" -> "캐모마일")
                key = name.replace(" ", "").replace("티", "").lower()
                # 메뉴명이 텍스트에 포함되어 있는지 확인
                if key in t:
                    print(f"[메뉴 파싱] 정확한 메뉴명 매칭 성공: {name} (key='{key}' in t='{t}')")
                    return cat, mid, name
            
            # 별칭 처리 (발음 변형 포함)
            result = _try_parse_menu_alias(cat, t)
            if result:
                return result
        
        # 지정된 카테고리에서 찾지 못했으면 전체 카테고리에서 검색
        print(f"[메뉴 파싱] 카테고리 '{category}'에서 찾지 못함, 전체 카테고리에서 검색 시작")
        categories_to_search = ["coffee", "ade", "tea", "dessert"]
    else:
        # category가 None이면 전체 카테고리에서 검색
        categories_to_search = ["coffee", "ade", "tea", "dessert"]
    
    # 2단계: 전체 카테고리에서 검색
    for cat in categories_to_search:
        # 정확한 메뉴명 매칭 (공백 제거 후 포함 여부 확인)
        for mid, name in _menu_choices_for_category(cat):
            # 메뉴명에서 공백과 "티" 제거 (예: "캐모마일 티" -> "캐모마일")
            key = name.replace(" ", "").replace("티", "").lower()
            # 메뉴명이 텍스트에 포함되어 있는지 확인
            if key in t:
                print(f"[메뉴 파싱] 정확한 메뉴명 매칭 성공: {name} (key='{key}' in t='{t}')")
                return cat, mid, name
        
        # 별칭 처리 (발음 변형 포함)
        result = _try_parse_menu_alias(cat, t)
        if result:
            return result
    
    return None


def _try_parse_menu_alias(cat: str, t: str) -> tuple[str, str, str] | None:
    """별칭 처리 (발음 변형 포함)"""
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
    elif cat == "ade":
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
    elif cat == "tea":
        # 캐모마일: 카모마일, 카모마일티, 카모, 캐모마일티, 캐모마일트, 캐모말, 캐모마, 케모마일, 카모메일
        if any(x in t for x in ["카모마일", "카모마일티", "카모", "캐모마일티", "캐모마일트", "캐모말", "캐모마", "케모마일", "카모메일", "캐모마일", "캐모"]):
            return "tea", "TEA_CHAMOMILE", "캐모마일 티"
        # 얼그레이: 얼그레이이, 얼그레, 얼그레잉, 얼그레잇, 얼그레에, 얼그레어, 얼그레오, 얼그레히, 얼글레이, 얼끌레이
        if any(x in t for x in ["얼그레이이", "얼그레", "얼그레잉", "얼그레잇", "얼그레에", "얼그레어", "얼그레오", "얼그레히", "얼글레이", "얼끌레이", "얼그레이", "얼그"]):
            return "tea", "TEA_EARL_GREY", "얼그레이 티"
        # 유자차: 유자챠, 유자차이, 유자차우, 유자타, 유자자, 유자티, 유자차
        if any(x in t for x in ["유자챠", "유자차이", "유자차우", "유자타", "유자자", "유자티", "유자차", "유자"]):
            return "tea", "TEA_YUJA", "유자차"
        # 녹차: 녹챠, 녹차이, 녹차우, 녹차어, 눅차, 녹타, 록차
        if any(x in t for x in ["녹챠", "녹차이", "녹차우", "녹차어", "눅차", "녹타", "록차", "녹차"]):
            return "tea", "TEA_GREEN", "녹차"
    elif cat == "dessert":
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
        # 마카롱: 마카론, 마까롱, 마카롱, 마카롱우, 마카롬, 마카롤, 마까론
        if any(x in t for x in ["마카론", "마까롱", "마카롱", "마카롱우", "마카롬", "마카롤", "마까론", "마카", "마까"]):
            return "dessert", "DESSERT_MACARON", "마카롱"
    
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
    # 작은사이즈 그대로 반환
    if "작은사이즈" in t or ("작은" in t and "사이즈" in t):
        return "작은사이즈"
    # 중간사이즈 그대로 반환
    if "중간사이즈" in t or ("중간" in t and "사이즈" in t):
        return "중간사이즈"
    # 큰사이즈 그대로 반환
    if "큰사이즈" in t or ("큰" in t and "사이즈" in t):
        return "큰사이즈"
    # 기존 인식 패턴
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
    # 쿠폰 체크는 다른 키워드보다 먼저 (쿠폰 사용할게 등)
    if "쿠폰" in t:
        return "coupon"
    if "카드" in t:
        return "card"
    if "현금" in t:
        return "cash"
    if "카카오페이" in t:
        return "kakaopay"
    if "페이" in t:
        return "pay"
    return None


# ───────────────────────────────────────────────
# LLM 기반 파싱 함수들
# ───────────────────────────────────────────────
def _parse_dine_type_llm(text: str) -> str | None:
    """LLM을 사용해 포장/매장 선택 의도 파싱"""
    DINE_TYPE_SYSTEM_PROMPT = """
    사용자 발화에서 포장/매장 선택 의도를 파싱하세요.
    
    가능한 값:
    - "takeout": 포장, 들고가기, 가져가기, 포장할게, 들고갈래, 포장해서 가져갈게, 테이크아웃, 들고가기로, 포장으로 등
    - "dinein": 매장, 먹고가기, 여기서 먹을래, 매장에서 먹을게, 여기서 먹을게, 먹고갈래, 여기서 드실래요, 매장에서, 여기서 등
    - null: 의도 파악 불가
    
    JSON 형식으로만 반환:
    {"dine_type": "takeout" | "dinein" | null}
    """
    
    DINE_TYPE_FEW_SHOTS = """
    예시 1)
    사용자: 포장할게
    응답: {"dine_type": "takeout"}
    
    예시 2)
    사용자: 매장에서 먹을래
    응답: {"dine_type": "dinein"}
    
    예시 3)
    사용자: 들고가기로 해줘
    응답: {"dine_type": "takeout"}
    
    예시 4)
    사용자: 여기서 먹고갈게요
    응답: {"dine_type": "dinein"}
    
    예시 5)
    사용자: 포장해서 가져갈게요
    응답: {"dine_type": "takeout"}
    
    예시 6)
    사용자: 매장에서 먹을래요
    응답: {"dine_type": "dinein"}
    """
    
    try:
        completion = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": DINE_TYPE_SYSTEM_PROMPT},
                {"role": "user", "content": DINE_TYPE_FEW_SHOTS},
                {"role": "user", "content": f"사용자: {text}\n응답:"},
            ],
            temperature=0.1,
            max_tokens=50,
        )
        
        raw = completion.choices[0].message.content.strip()
        
        # 마크다운 코드 블록 제거
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
            raw = raw.strip().rstrip("```").strip()
        
        print(f"[_parse_dine_type_llm] LLM raw 응답: {raw}")
        
        data = json.loads(raw)
        dine_type = data.get("dine_type")
        print(f"[_parse_dine_type_llm] 파싱된 dine_type: {dine_type}")
        return dine_type
    except Exception as e:
        print(f"[_parse_dine_type_llm] 오류: {e}")
        return None


def _parse_menu_item_llm(text: str, category: str | None) -> tuple[str, str, str] | None:
    """LLM을 사용해 메뉴 선택 의도 파싱"""
    # 모든 메뉴 목록 생성
    all_menus = []
    for cat in ["coffee", "ade", "tea", "dessert"]:
        for menu_id, menu_name in _menu_choices_for_category(cat):
            all_menus.append(f"- {cat}: {menu_id} ({menu_name})")
    
    menu_list = "\n".join(all_menus)
    
    MENU_SYSTEM_PROMPT = f"""
    사용자 발화에서 메뉴 선택 의도를 파싱하세요.
    
    가능한 메뉴 목록:
    {menu_list}
    
    사용자가 메뉴를 선택하려는 의도인지 판단하고, 메뉴명을 추출하세요.
    UI 위치 질문("어디있어", "어딨어" 등)이 아닌 메뉴 주문 의도만 처리하세요.
    
    JSON 형식으로 반환:
    {{
        "category": "coffee" | "ade" | "tea" | "dessert" | null,
        "menu_id": "COFFEE_AMERICANO" | ... | null,
        "menu_name": "아메리카노" | ... | null
    }}
    
    - 메뉴를 찾으면 category, menu_id, menu_name 모두 반환
    - 찾지 못하면 모두 null 반환
    """
    
    MENU_FEW_SHOTS = """
    예시 1)
    사용자: 아메리카노 먹을래
    응답: {"category": "coffee", "menu_id": "COFFEE_AMERICANO", "menu_name": "아메리카노"}
    
    예시 2)
    사용자: 아메리카노 선택할게
    응답: {"category": "coffee", "menu_id": "COFFEE_AMERICANO", "menu_name": "아메리카노"}
    
    예시 3)
    사용자: 아메리카노로 줘
    응답: {"category": "coffee", "menu_id": "COFFEE_AMERICANO", "menu_name": "아메리카노"}
    
    예시 4)
    사용자: 레몬에이드 주문할래
    응답: {"category": "ade", "menu_id": "ADE_LEMON", "menu_name": "레몬에이드"}
    
    예시 5)
    사용자: 치즈케이크 주세요
    응답: {"category": "dessert", "menu_id": "DESSERT_CHEESECAKE", "menu_name": "치즈케이크"}
    
    예시 6)
    사용자: 아메리카노 하나 주세요
    응답: {"category": "coffee", "menu_id": "COFFEE_AMERICANO", "menu_name": "아메리카노"}
    
    예시 7)
    사용자: 아메리카노 시킬게
    응답: {"category": "coffee", "menu_id": "COFFEE_AMERICANO", "menu_name": "아메리카노"}
    """
    
    context = f"현재 지정된 카테고리: {category or '없음'}\n" if category else ""
    
    try:
        completion = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": MENU_SYSTEM_PROMPT},
                {"role": "user", "content": MENU_FEW_SHOTS},
                {"role": "user", "content": f"{context}사용자: {text}\n응답:"},
            ],
            temperature=0.1,
            max_tokens=100,
        )
        
        raw = completion.choices[0].message.content.strip()
        
        # 마크다운 코드 블록 제거
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
            raw = raw.strip().rstrip("```").strip()
        
        print(f"[_parse_menu_item_llm] LLM raw 응답: {raw}")
        
        data = json.loads(raw)
        category_val = data.get("category")
        menu_id = data.get("menu_id")
        menu_name = data.get("menu_name")
        
        if category_val and menu_id and menu_name:
            print(f"[_parse_menu_item_llm] 파싱 성공: category={category_val}, menu_id={menu_id}, menu_name={menu_name}")
            return (category_val, menu_id, menu_name)
    except Exception as e:
        print(f"[_parse_menu_item_llm] 오류: {e}")
    
    return None


def _parse_cart_action_llm(text: str) -> dict | None:
    """LLM을 사용해 장바구니 복합 액션(제거+추가) 파싱"""
    # 모든 메뉴 목록 생성
    all_menus = []
    for cat in ["coffee", "ade", "tea", "dessert"]:
        for menu_id, menu_name in _menu_choices_for_category(cat):
            all_menus.append(f"- {cat}: {menu_id} ({menu_name})")
    
    menu_list = "\n".join(all_menus)
    
    CART_ACTION_SYSTEM_PROMPT = f"""
    사용자 발화에서 장바구니 제거 및 추가 액션을 파싱하세요.
    
    가능한 메뉴 목록:
    {menu_list}
    
    "치즈케이크 빼고 마카롱 담아줘"처럼 제거할 메뉴와 추가할 메뉴가 함께 있는 경우를 처리하세요.
    
    JSON 형식으로 반환:
    {{
        "remove_menu": {{
            "category": "dessert" | null,
            "menu_id": "DESSERT_CHEESECAKE" | null,
            "menu_name": "치즈케이크" | null
        }},
        "add_menu": {{
            "category": "dessert" | null,
            "menu_id": "DESSERT_MACARON" | null,
            "menu_name": "마카롱" | null
        }}
    }}
    
    - 제거할 메뉴만 있으면 remove_menu만 채우고 add_menu는 모두 null
    - 추가할 메뉴만 있으면 add_menu만 채우고 remove_menu는 모두 null
    - 둘 다 있으면 둘 다 채움
    - "티라미수 빼줘", "치즈케이크 제거해줘"처럼 "장바구니" 키워드가 없어도 "빼", "빼줘", "제거" 등의 키워드가 있으면 제거 의도로 판단
    - 메뉴 목록에 없는 메뉴도 사용자가 말했다면 menu_name만 추출하고, category와 menu_id는 null로 반환
    - 찾지 못하면 null 반환
    
    중요: 메뉴 목록에 정확히 일치하는 메뉴가 있으면 정확한 menu_id를 반환하되, 없어도 사용자가 말한 메뉴 이름(menu_name)은 추출하세요.
    """
    
    CART_ACTION_FEW_SHOTS = """
    예시 1)
    사용자: 치즈케이크 빼고 마카롱 담아줘
    응답: {"remove_menu": {"category": "dessert", "menu_id": "DESSERT_CHEESECAKE", "menu_name": "치즈케이크"}, "add_menu": {"category": "dessert", "menu_id": "DESSERT_MACARON", "menu_name": "마카롱"}}
    
    예시 2)
    사용자: 티라미수 빼고 아메리카노 담아줘
    응답: {"remove_menu": {"category": "dessert", "menu_id": "DESSERT_TIRAMISU", "menu_name": "티라미수"}, "add_menu": {"category": "coffee", "menu_id": "COFFEE_AMERICANO", "menu_name": "아메리카노"}}
    
    예시 3)
    사용자: 치즈케이크 담아줘
    응답: {"remove_menu": {"category": null, "menu_id": null, "menu_name": null}, "add_menu": {"category": "dessert", "menu_id": "DESSERT_CHEESECAKE", "menu_name": "치즈케이크"}}
    
    예시 4)
    사용자: 티라미수 장바구니에서 빼줘
    응답: {"remove_menu": {"category": "dessert", "menu_id": "DESSERT_TIRAMISU", "menu_name": "티라미수"}, "add_menu": {"category": null, "menu_id": null, "menu_name": null}}
    
    예시 5)
    사용자: 티라미수 빼줘
    응답: {"remove_menu": {"category": "dessert", "menu_id": "DESSERT_TIRAMISU", "menu_name": "티라미수"}, "add_menu": {"category": null, "menu_id": null, "menu_name": null}}
    
    예시 6)
    사용자: 치즈케이크 제거해줘
    응답: {"remove_menu": {"category": "dessert", "menu_id": "DESSERT_CHEESECAKE", "menu_name": "치즈케이크"}, "add_menu": {"category": null, "menu_id": null, "menu_name": null}}
    """
    
    try:
        completion = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CART_ACTION_SYSTEM_PROMPT},
                {"role": "user", "content": CART_ACTION_FEW_SHOTS},
                {"role": "user", "content": f"사용자: {text}\n응답:"},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        
        raw = completion.choices[0].message.content.strip()
        
        # 마크다운 코드 블록 제거
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
            raw = raw.strip().rstrip("```").strip()
        
        print(f"[_parse_cart_action_llm] LLM raw 응답: {raw}")
        
        data = json.loads(raw)
        return data
    except Exception as e:
        print(f"[_parse_cart_action_llm] 오류: {e}")
    
    return None


def _parse_temp_llm(text: str) -> str | None:
    """LLM을 사용해 온도 선택 의도 파싱"""
    TEMP_SYSTEM_PROMPT = """
    사용자 발화에서 온도 선택 의도를 파싱하세요.
    
    가능한 값:
    - "hot": 따뜻하게, 뜨겁게, 핫, 따뜻하게 주세요, 뜨겁게 해줘, 따뜻한 걸로, 핫으로 등
    - "ice": 차갑게, 아이스, 차갑게 주세요, 아이스로 해줘, 차가운 걸로, 아이스로, 시원하게 등
    - null: 의도 파악 불가
    
    JSON 형식으로만 반환:
    {"temp": "hot" | "ice" | null}
    """
    
    TEMP_FEW_SHOTS = """
    예시 1)
    사용자: 따뜻하게 주세요
    응답: {"temp": "hot"}
    
    예시 2)
    사용자: 아이스로 해줘
    응답: {"temp": "ice"}
    
    예시 3)
    사용자: 차갑게 할게
    응답: {"temp": "ice"}
    
    예시 4)
    사용자: 뜨겁게 주세요
    응답: {"temp": "hot"}
    
    예시 5)
    사용자: 핫으로 해줘
    응답: {"temp": "hot"}
    """
    
    try:
        completion = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": TEMP_SYSTEM_PROMPT},
                {"role": "user", "content": TEMP_FEW_SHOTS},
                {"role": "user", "content": f"사용자: {text}\n응답:"},
            ],
            temperature=0.1,
            max_tokens=50,
        )
        
        raw = completion.choices[0].message.content.strip()
        
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
            raw = raw.strip().rstrip("```").strip()
        
        print(f"[_parse_temp_llm] LLM raw 응답: {raw}")
        
        data = json.loads(raw)
        temp = data.get("temp")
        print(f"[_parse_temp_llm] 파싱된 temp: {temp}")
        return temp
    except Exception as e:
        print(f"[_parse_temp_llm] 오류: {e}")
        return None


def _parse_size_llm(text: str) -> str | None:
    """LLM을 사용해 사이즈 선택 의도 파싱"""
    SIZE_SYSTEM_PROMPT = """
    사용자 발화에서 사이즈 선택 의도를 파싱하세요.
    
    가능한 값:
    - "작은사이즈": 작은, 작은 사이즈, 스몰, 톨, 작은 걸로, 작은 사이즈로 등
    - "중간사이즈": 중간, 중간 사이즈, 미디엄, 그란데, 중간 걸로, 중간 사이즈로, 보통 등
    - "큰사이즈": 큰, 큰 사이즈, 라지, 벤티, 큰 걸로, 큰 사이즈로 등
    
    JSON 형식으로만 반환:
    {"size": "작은사이즈" | "중간사이즈" | "큰사이즈" | null}
    """
    
    SIZE_FEW_SHOTS = """
    예시 1)
    사용자: 작은 사이즈 주세요
    응답: {"size": "작은사이즈"}
    
    예시 2)
    사용자: 중간 사이즈로 해줘
    응답: {"size": "중간사이즈"}
    
    예시 3)
    사용자: 큰 사이즈 할게
    응답: {"size": "큰사이즈"}
    
    예시 4)
    사용자: 그란데로 주세요
    응답: {"size": "중간사이즈"}
    
    예시 5)
    사용자: 스몰 사이즈
    응답: {"size": "작은사이즈"}
    
    예시 6)
    사용자: 벤티로 해줘
    응답: {"size": "큰사이즈"}
    """
    
    try:
        completion = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SIZE_SYSTEM_PROMPT},
                {"role": "user", "content": SIZE_FEW_SHOTS},
                {"role": "user", "content": f"사용자: {text}\n응답:"},
            ],
            temperature=0.1,
            max_tokens=50,
        )
        
        raw = completion.choices[0].message.content.strip()
        
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
            raw = raw.strip().rstrip("```").strip()
        
        print(f"[_parse_size_llm] LLM raw 응답: {raw}")
        
        data = json.loads(raw)
        size = data.get("size")
        print(f"[_parse_size_llm] 파싱된 size: {size}")
        return size
    except Exception as e:
        print(f"[_parse_size_llm] 오류: {e}")
        return None


def _parse_options_llm(category: str, text: str, options: dict) -> dict:
    """LLM을 사용해 옵션 선택 의도 파싱"""
    OPTIONS_SYSTEM_PROMPT = f"""
    사용자 발화에서 옵션 선택 의도를 파싱하세요.
    
    현재 카테고리: {category}
    
    {category == "coffee" and """
    커피 옵션:
    - 디카페인: "decaf": true
    - 시럽 추가: "syrup": true
    - 샷 추가: "extra_shot": 0 (기본), 1, 2, 3 등
    """ or ""}
    
    {category == "ade" and """
    에이드 옵션:
    - 당도 조절: "sweetness": "low" | "normal" | "high"
    """ or ""}
    
    기존 옵션: {json.dumps(options, ensure_ascii=False)}
    
    사용자가 선택한 옵션만 반영하세요. 언급하지 않은 옵션은 기존 값을 유지하세요.
    
    JSON 형식으로 반환 (전체 options 객체):
    {json.dumps({
        "extra_shot": 0,
        "syrup": False,
        "decaf": None,
        "sweetness": None
    } if category == "coffee" else {
        "extra_shot": 0,
        "syrup": False,
        "decaf": None,
        "sweetness": "normal"
    }, ensure_ascii=False)}
    """
    
    OPTIONS_FEW_SHOTS_COFFEE = """
    예시 1)
    사용자: 디카페인으로 해줘
    기존: {"extra_shot": 0, "syrup": False, "decaf": None, "sweetness": None}
    응답: {"extra_shot": 0, "syrup": False, "decaf": True, "sweetness": None}
    
    예시 2)
    사용자: 시럽 추가해줘
    기존: {"extra_shot": 0, "syrup": False, "decaf": None, "sweetness": None}
    응답: {"extra_shot": 0, "syrup": True, "decaf": None, "sweetness": None}
    
    예시 3)
    사용자: 샷 하나 추가해줘
    기존: {"extra_shot": 0, "syrup": False, "decaf": None, "sweetness": None}
    응답: {"extra_shot": 1, "syrup": False, "decaf": None, "sweetness": None}
    
    예시 4)
    사용자: 샷 두 개 추가
    기존: {"extra_shot": 0, "syrup": False, "decaf": None, "sweetness": None}
    응답: {"extra_shot": 2, "syrup": False, "decaf": None, "sweetness": None}
    """
    
    OPTIONS_FEW_SHOTS_ADE = """
    예시 1)
    사용자: 연하게 해줘
    기존: {"extra_shot": 0, "syrup": False, "decaf": None, "sweetness": "normal"}
    응답: {"extra_shot": 0, "syrup": False, "decaf": None, "sweetness": "low"}
    
    예시 2)
    사용자: 달게 해줘
    기존: {"extra_shot": 0, "syrup": False, "decaf": None, "sweetness": "normal"}
    응답: {"extra_shot": 0, "syrup": False, "decaf": None, "sweetness": "high"}
    
    예시 3)
    사용자: 보통으로 해줘
    기존: {"extra_shot": 0, "syrup": False, "decaf": None, "sweetness": "normal"}
    응답: {"extra_shot": 0, "syrup": False, "decaf": None, "sweetness": "normal"}
    """
    
    few_shots = OPTIONS_FEW_SHOTS_COFFEE if category == "coffee" else OPTIONS_FEW_SHOTS_ADE
    
    try:
        completion = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": OPTIONS_SYSTEM_PROMPT},
                {"role": "user", "content": few_shots},
                {"role": "user", "content": f"기존: {json.dumps(options, ensure_ascii=False)}\n사용자: {text}\n응답:"},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        
        raw = completion.choices[0].message.content.strip()
        
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
            raw = raw.strip().rstrip("```").strip()
        
        print(f"[_parse_options_llm] LLM raw 응답: {raw}")
        
        data = json.loads(raw)
        print(f"[_parse_options_llm] 파싱된 options: {data}")
        # 파싱된 데이터가 유효한지 확인 (dict이고 필수 필드 존재)
        if isinstance(data, dict) and "extra_shot" in data:
            return data
        else:
            print(f"[_parse_options_llm] 유효하지 않은 데이터, 기존 options 반환")
            return options
    except Exception as e:
        print(f"[_parse_options_llm] 오류: {e}, 기존 options 반환")
        return options


def _parse_payment_llm(text: str) -> str | None:
    """LLM을 사용해 결제 수단 선택 의도 파싱"""
    PAYMENT_SYSTEM_PROMPT = """
    사용자 발화에서 결제 수단 선택 의도를 파싱하세요.
    
    가능한 값:
    - "card": 카드, 카드결제, 카드로, 카드로 결제할게, 카드 결제해줘, 신용카드, 카드로 할게 등
    - "cash": 현금, 현금으로, 현금 결제, 현금으로 할게 등
    - "kakaopay": 카카오페이, 카카오페이로, 카카오페이 결제 등
    - "coupon": 쿠폰, 쿠폰으로, 쿠폰 결제, 쿠폰으로 할게, 쿠폰 사용할게, 쿠폰 사용할래, 쿠폰으로 결제할게 등
    - "pay": 간편결제, 페이, 페이로 (구체적 수단 불명확)
    - null: 의도 파악 불가
    
    JSON 형식으로만 반환:
    {"payment_method": "card" | "cash" | "kakaopay" | "coupon" | "pay" | null}
    """
    
    PAYMENT_FEW_SHOTS = """
    예시 1)
    사용자: 카드결제할게
    응답: {"payment_method": "card"}
    
    예시 2)
    사용자: 카드로 결제해줘
    응답: {"payment_method": "card"}
    
    예시 3)
    사용자: 쿠폰으로 할게요
    응답: {"payment_method": "coupon"}
    
    예시 4)
    사용자: 쿠폰 사용할게
    응답: {"payment_method": "coupon"}
    
    예시 5)
    사용자: 쿠폰 결제할게
    응답: {"payment_method": "coupon"}
    
    예시 6)
    사용자: 쿠폰 사용할래
    응답: {"payment_method": "coupon"}
    
    예시 7)
    사용자: 카카오페이로 주세요
    응답: {"payment_method": "kakaopay"}
    
    예시 8)
    사용자: 현금 결제할래
    응답: {"payment_method": "cash"}
    
    예시 9)
    사용자: 카드로 할게
    응답: {"payment_method": "card"}
    """
    
    try:
        completion = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PAYMENT_SYSTEM_PROMPT},
                {"role": "user", "content": PAYMENT_FEW_SHOTS},
                {"role": "user", "content": f"사용자: {text}\n응답:"},
            ],
            temperature=0.1,
            max_tokens=50,
        )
        
        raw = completion.choices[0].message.content.strip()
        
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
            raw = raw.strip().rstrip("```").strip()
        
        print(f"[_parse_payment_llm] LLM raw 응답: {raw}")
        
        data = json.loads(raw)
        payment_method = data.get("payment_method")
        print(f"[_parse_payment_llm] 파싱된 payment_method: {payment_method}")
        return payment_method
    except Exception as e:
        print(f"[_parse_payment_llm] 오류: {e}")
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
    add_to_cart = ctx.get("add_to_cart", False)  # 장바구니 추가 플래그
    remove_from_cart = ctx.get("remove_from_cart", False)  # 장바구니 제거 플래그

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

    payload = {
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
    
    # 장바구니 추가 플래그가 설정되어 있으면 추가하고 초기화
    if add_to_cart:
        payload["add_to_cart"] = True
        ctx["add_to_cart"] = False  # 사용 후 초기화
    
    # 장바구니 제거 플래그가 설정되어 있으면 추가하고 초기화
    if remove_from_cart:
        payload["remove_from_cart"] = True
        # 제거할 메뉴 정보가 별도로 저장되어 있으면 포함
        if ctx.get("remove_menu_category") and ctx.get("remove_menu_id") and ctx.get("remove_menu_name"):
            payload["remove_menu"] = {
                "category": ctx.get("remove_menu_category"),
                "menu_id": ctx.get("remove_menu_id"),
                "menu_name": ctx.get("remove_menu_name"),
            }
            # 사용 후 초기화
            ctx["remove_menu_category"] = None
            ctx["remove_menu_id"] = None
            ctx["remove_menu_name"] = None
        ctx["remove_from_cart"] = False  # 사용 후 초기화
    
    return payload


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

메뉴 아이템 (메뉴 리스트 화면)
- menu_item_coffee_americano (아메리카노)
- menu_item_coffee_espresso (에스프레소)
- menu_item_coffee_latte (카페 라떼)
- menu_item_coffee_cappuccino (카푸치노)
- menu_item_ade_lemon (레몬에이드)
- menu_item_ade_grapefruit (자몽에이드)
- menu_item_ade_green_grape (청포도 에이드)
- menu_item_ade_orange (오렌지 에이드)
- menu_item_tea_chamomile (캐모마일 티)
- menu_item_tea_earl_grey (얼그레이 티)
- menu_item_tea_yuja (유자차)
- menu_item_tea_green (녹차)
- menu_item_dessert_cheesecake (치즈케이크)
- menu_item_dessert_tiramisu (티라미수)
- menu_item_dessert_brownie (초코 브라우니)
- menu_item_dessert_croissant (크루아상)
- menu_item_dessert_macaron (마카롱)

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
- 사용자가 특정 메뉴 아이템(예: "아메리카노", "유자차", "레몬에이드")의 위치를 물어보면, 반드시 해당 메뉴 아이템의 target_element_id를 사용하라.
- 예: "아메리카노 어딨어?" → target_element_id: "menu_item_coffee_americano"
- 예: "유자차는 어디 있나요?" → target_element_id: "menu_item_tea_yuja"
- 예: "레몬에이드 어디에 있어?" → target_element_id: "menu_item_ade_lemon"
- "이전으로 갈려면 어떻게 해?", "뒤로 가려면 뭐 눌러야 해?" 같은 질문에서는 현재 대화 단계에 맞는 prev 버튼의 target_element_id를 반환하라.
  - temp 단계: temp_prev_button
  - size 단계: size_prev_button
  - options 단계: option_prev_button
  - confirm/payment 단계: payment_prev_button
  - ⚠️ 중요: 이전 버튼은 절대 "상단"이 아니라 항상 "왼쪽 하단"에 위치한다. answer_text는 반드시 "지금 키오스크 왼쪽 하단에 있는 이전으로 버튼을 눌러주시면 됩니다." 형식으로만 작성하라.
- "다음으로 가려면 어떻게 해?" 같은 질문에서는 현재 대화 단계에 맞는 next 버튼의 target_element_id를 반환하라.
  - temp 단계: temp_next_button
  - size 단계: size_next_button
  - options 단계: option_next_button
  - 다음 버튼의 answer_text는 "화면 오른쪽에 있는 '다음' 버튼을 눌러주세요." 형식으로 작성하라.
- 모르면 target_element_id에는 null을 넣고,
  answer_text는 "어느 버튼을 찾으시는지 다시 한번 말씀해 주세요."라고 해라.
- answer_text는 노인이 이해하기 쉽게, 존댓말로, 1~2문장으로 안내해라.
- 반드시 아래 JSON 형식으로만 출력해라. 다른 텍스트는 절대 쓰지 마라.
- target_element_id는 반드시 포함해야 하며, 메뉴 아이템을 물어보면 null이 아닌 적절한 값을 반환해야 한다.

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
  "answer_text": "메뉴 선택을 다 하셨으면, 화면 오른쪽 아래 파란색 '결제하기' 버튼을 눌러 주세요."
}

예시 2)
사용자: 장바구니는 어디 있어?
응답:
{
  "target_element_id": "menu_cart_area",
  "answer_text": "화면 아래쪽 가운데에 있는 '장바구니' 영역에서 주문하신 메뉴를 보실 수 있습니다."
}

예시 3)
사용자: 처음으로 돌아가는 거 어디야?
응답:
{
  "target_element_id": "menu_home_button",
  "answer_text": "화면 오른쪽 상단에 있는 동그란 '홈' 버튼을 눌러 주세요."
}

예시 4)
사용자: 아메리카노 어딨어?
응답:
{
  "target_element_id": "menu_item_coffee_americano",
  "answer_text": "아메리카노는 메뉴판 상단 커피 섹션에 있습니다."
}

예시 5)
사용자: 유자차는 어디 있나요?
응답:
{
  "target_element_id": "menu_item_tea_yuja",
  "answer_text": "유자차는 메뉴판 차 섹션에 있습니다."
}

예시 6)
사용자: 레몬에이드 어디에 있어?
응답:
{
  "target_element_id": "menu_item_ade_lemon",
  "answer_text": "레몬에이드는 메뉴판 에이드 섹션에 있습니다."
}

예시 7)
사용자: 이전으로 갈려면 어떻게 해?
현재 대화 단계: temp
응답:
{
  "target_element_id": "temp_prev_button",
  "answer_text": "지금 키오스크 왼쪽 하단에 있는 이전으로 버튼을 눌러주시면 됩니다."
}

예시 8)
사용자: 뒤로 가려면 뭐 눌러야 해?
현재 대화 단계: size
응답:
{
  "target_element_id": "size_prev_button",
  "answer_text": "지금 키오스크 왼쪽 하단에 있는 이전으로 버튼을 눌러주시면 됩니다."
}

예시 9)
사용자: 다음으로 가려면 어떻게 해?
현재 대화 단계: temp
응답:
{
  "target_element_id": "temp_next_button",
  "answer_text": "화면 오른쪽에 있는 '다음' 버튼을 눌러주세요."
}

예시 10)
사용자: 이전으로 가려면 어떻게해?
현재 대화 단계: size
응답:
{
  "target_element_id": "size_prev_button",
  "answer_text": "지금 키오스크 왼쪽 하단에 있는 이전으로 버튼을 눌러주시면 됩니다."
}

예시 11)
사용자: 다음으로 갈려면 어떻게해?
현재 대화 단계: size
응답:
{
  "target_element_id": "size_next_button",
  "answer_text": "화면 오른쪽에 있는 '다음' 버튼을 눌러주세요."
}
""".strip()


def _menu_id_to_target_element_id(menu_id: str) -> str | None:
    """
    메뉴 ID를 target_element_id로 변환.
    예: "COFFEE_AMERICANO" -> "menu_item_coffee_americano"
    """
    mapping = {
        "COFFEE_AMERICANO": "menu_item_coffee_americano",
        "COFFEE_ESPRESSO": "menu_item_coffee_espresso",
        "COFFEE_LATTE": "menu_item_coffee_latte",
        "COFFEE_CAPPUCCINO": "menu_item_coffee_cappuccino",
        "ADE_LEMON": "menu_item_ade_lemon",
        "ADE_GRAPEFRUIT": "menu_item_ade_grapefruit",
        "ADE_GREEN_GRAPE": "menu_item_ade_green_grape",
        "ADE_ORANGE": "menu_item_ade_orange",
        "TEA_CHAMOMILE": "menu_item_tea_chamomile",
        "TEA_EARL_GREY": "menu_item_tea_earl_grey",
        "TEA_YUJA": "menu_item_tea_yuja",
        "TEA_GREEN": "menu_item_tea_green",
        "DESSERT_CHEESECAKE": "menu_item_dessert_cheesecake",
        "DESSERT_TIRAMISU": "menu_item_dessert_tiramisu",
        "DESSERT_BROWNIE": "menu_item_dessert_brownie",
        "DESSERT_CROISSANT": "menu_item_dessert_croissant",
        "DESSERT_MACARON": "menu_item_dessert_macaron",
    }
    return mapping.get(menu_id)


def looks_like_ui_help(text: str) -> bool:
    """
    화면에서 버튼/영역 위치를 묻는 발화인지 간단 키워드로 감지.
    단, 메뉴명이 포함된 경우(예: "아메리카노 장바구니에 담아줘")는 False 반환.
    단, 결제 의도가 명확한 경우(예: "결제하기", "결제할게요")는 False 반환.
    단, 위치 질문 키워드("어디", "어딨어")가 있으면 메뉴명이 있어도 UI 도움말로 처리.
    """
    t = text.replace(" ", "").lower()
    
    # 위치 질문 키워드가 있으면 메뉴명이 있어도 UI 도움말로 처리
    location_question_keywords = ["어디", "어딨어", "어디있", "어디있어", "어디에", "어디에있", "있어", "있나", "있는지", "있어요", "있나요"]
    if any(keyword in t for keyword in location_question_keywords):
        return True  # 위치 질문이면 무조건 UI 도움말
    
    # 결제 의도가 명확한 경우 (예: "결제하기", "결제할게요")는 UI 도움말이 아님
    payment_intent_keywords = [
        "결제하기", "결제할게요", "결제하겠어요", "결제하겠습니다", 
        "결제할게", "결제하자", "결제해줘", "결제해주세요"
    ]
    if any(keyword in t for keyword in payment_intent_keywords):
        return False  # 결제 의도가 명확하면 UI 도움말이 아님
    
    # 이전/뒤로 버튼 위치를 물어보는 경우는 UI 도움말로 처리
    # 예: "이전으로 갈려면 어떻게 해?", "뒤로 가려면 뭐 눌러야 해?" 등
    question_keywords = ["어떻게", "어디", "뭐", "무엇", "방법", "어떡해", "어떻게해", "뭐눌러", "뭐눌러야"]
    back_button_keywords = ["이전", "뒤로", "돌아가", "이전으로", "뒤로가", "돌아가기"]
    if any(q in t for q in question_keywords) and any(b in t for b in back_button_keywords):
        return True  # 이전/뒤로 버튼 위치 질문은 UI 도움말
    
    # 단순 액션 의도("이전", "뒤로"만 있는 경우)는 UI 도움말이 아님
    # 이들은 규칙 기반으로 각 step에서 처리됨
    simple_back_keywords = ["이전", "뒤로", "돌아가", "취소", "back", "prev"]
    if any(keyword in t for keyword in simple_back_keywords) and not any(q in t for q in question_keywords):
        return False  # 단순 이전 액션이면 UI 도움말이 아님
    
    # 메뉴명이 포함되어 있고 액션 키워드("담아줘", "주세요" 등)가 있으면 메뉴 선택 의도
    menu_keywords = [
        "아메리카노", "아메", "에스프레소", "라떼", "카푸치노", "카푸",
        "레몬에이드", "레몬", "자몽에이드", "자몽", "청포도에이드", "청포도", "오렌지에이드", "오렌지",
        "캐모마일", "얼그레이", "유자차", "유자", "녹차",
        "치즈케이크", "티라미수", "브라우니", "크루아상"
    ]
    action_keywords = ["담아", "담아줘", "주세요", "주문", "하나", "한잔", "추가"]
    
    # 메뉴명이 있고 액션 키워드도 있으면 메뉴 선택 의도 (UI 도움말 아님)
    has_menu = any(menu in t for menu in menu_keywords)
    has_action = any(action in t for action in action_keywords)
    if has_menu and has_action:
        return False  # 메뉴명 + 액션 = 메뉴 선택 의도
    
    # 다음 버튼 위치를 물어보는 경우도 UI 도움말로 처리
    next_button_keywords = ["다음", "다음으로", "다음단계", "계속"]
    if any(q in t for q in question_keywords) and any(n in t for n in next_button_keywords):
        return True  # 다음 버튼 위치 질문은 UI 도움말
    
    # UI 도움말 키워드 체크
    keywords = [
        "버튼", "어디", "어딨어", "다음", "홈",
        "장바구니", "결제", "처음으로", "전송", "qr", "큐알"
    ]
    return any(k in t for k in keywords)


def classify_ui_target(user_text: str, current_step: str | None = None) -> dict:
    """
    OpenAI에 UI용 프롬프트로 물어보고
    { "target_element_id": ..., "answer_text": ... } 형태로 반환.
    
    Args:
        user_text: 사용자 발화 텍스트
        current_step: 현재 대화 단계 (선택적, 이전/다음 버튼 판단에 사용)
    """
    # 현재 step 정보를 프롬프트에 포함
    step_context = ""
    if current_step:
        step_context = f"\n현재 대화 단계: {current_step}\n"
        if current_step == "temp":
            step_context += "이 단계에서는 temp_prev_button, temp_next_button을 사용하세요."
        elif current_step == "size":
            step_context += "이 단계에서는 size_prev_button, size_next_button을 사용하세요."
        elif current_step == "options":
            step_context += "이 단계에서는 option_prev_button, option_next_button을 사용하세요."
        elif current_step in ["confirm", "payment"]:
            step_context += "이 단계에서는 payment_prev_button을 사용하세요."
    
    user_prompt = f"사용자: {user_text}{step_context}\n응답:"
    
    completion = gpt_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": UI_SYSTEM_PROMPT},
            {"role": "user", "content": UI_FEW_SHOTS},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=150,
    )

    raw = completion.choices[0].message.content.strip()
    
    # 마크다운 코드 블록 제거 (```json ... ``` 형식)
    if raw.startswith("```"):
        # 첫 번째 ``` 이후부터 마지막 ``` 이전까지 추출
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
        # 마지막 ``` 제거
        raw = raw.strip().rstrip("```").strip()
    
    # 디버깅을 위한 로깅
    print(f"[classify_ui_target] LLM raw 응답: {raw}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[classify_ui_target] JSON 파싱 실패: {e}, raw: {raw}")
        # JSON 파싱 실패 시에도 텍스트에서 target_element_id 찾기 시도
        data = {
            "target_element_id": None,
            "answer_text": "어느 버튼을 찾으시는지 다시 한번 말씀해 주세요."
        }
        
        # raw 텍스트에서 "menu_item_" 패턴 찾기
        import re
        menu_item_match = re.search(r'"menu_item_\w+"', raw)
        if menu_item_match:
            data["target_element_id"] = menu_item_match.group(0).strip('"')
            print(f"[classify_ui_target] 텍스트에서 추출한 target_element_id: {data['target_element_id']}")

    # 방어적 필드 정리
    if "target_element_id" not in data:
        data["target_element_id"] = None
    if "answer_text" not in data:
        data["answer_text"] = "어느 버튼을 찾으시는지 다시 한번 말씀해 주세요."
    
    print(f"[classify_ui_target] 파싱된 데이터: {data}")

    return data


# ───────────────────────────────────────────────
# OpenAI helper mode (대화형 자유 질문 답변)
# ───────────────────────────────────────────────
def looks_like_general_question(text: str) -> bool:
    """
    사용자가 메뉴/단계 외 일반 질문을 하는 상황 감지.
    예: '현금 돼?', '현금으로도 결제 돼?', '텍스트 크기 키워줘'
    (UI 위치 질문은 looks_like_ui_help가 먼저 처리함)
    """
    t = text.strip().lower()

    # 텍스트 크기 관련 요청
    if re.search(r"(텍스트|글자|글씨|폰트).*크기.*(키워|크게|늘려|줄여|작게|리셋|원래|초기화|되돌리)", t) or \
       re.search(r"(텍스트|글자|글씨|폰트).*(키워|크게|늘려|줄여|작게|리셋|원래|초기화|되돌리)", t):
        return True

    # 바코드 관련 질문
    if re.search(r"바코드.*(어떻게|방법|인식|스캔)", t) or \
       re.search(r"(바코드|qr|큐알).*(어떻게|방법|인식|스캔)", t):
        return True

    # 결제 관련 질문
    if re.search(r"(현금|카드|결제)\s*(되|가능|돼)", t):
        return True

    # 안내 요청
    if "어떻게" in t or "방법" in t:
        return True

    # '메뉴 추천해줘', '뭐가 맛있어?' 등
    if re.search(r"(추천|맛있|뭐먹|뭐가)", t):
        return True

    # '?' 체크는 제거 - UI 위치 질문과 구분하기 위해
    # UI 위치 질문은 looks_like_ui_help()가 먼저 처리함

    return False


def answer_general_question(text: str) -> tuple[str, str | None]:
    """
    OpenAI API를 사용해 kiosk 안내 톤으로 대답 생성.
    특정 요청(텍스트 크기 등)은 규칙 기반으로 처리.
    
    Returns:
        (response_text, ui_action): 응답 텍스트와 UI 액션 (없으면 None)
    """
    t = text.strip().lower()
    
    # 텍스트 크기 관련 요청 처리
    if re.search(r"(텍스트|글자|글씨|폰트).*크기.*(키워|크게|늘려)", t) or \
       re.search(r"(텍스트|글자|글씨|폰트).*(키워|크게|늘려)", t):
        return "텍스트 크기를 키워드리겠습니다.", "text_size_increase"
    
    if re.search(r"(텍스트|글자|글씨|폰트).*크기.*(줄여|작게)", t) or \
       re.search(r"(텍스트|글자|글씨|폰트).*(줄여|작게)", t):
        return "텍스트 크기를 줄여드리겠습니다.", "text_size_decrease"
    
    # 텍스트 크기 리셋 처리
    if re.search(r"(텍스트|글자|글씨|폰트).*크기.*(리셋|원래|초기화|되돌리)", t) or \
       re.search(r"(텍스트|글자|글씨|폰트).*(리셋|원래|초기화|되돌리)", t):
        return "텍스트 크기를 원래 크기로 되돌리겠습니다.", "text_size_reset"
    
    # 바코드 인식 방법 안내
    if re.search(r"바코드.*(어떻게|방법|인식|스캔)", t) or \
       re.search(r"(바코드|qr|큐알).*(어떻게|방법|인식|스캔)", t):
        return "아래 바코드기에 핸드폰을 대고 인식시켜주세요.", None
    
    # 그 외는 OpenAI로 답변 생성
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

    return completion.choices[0].message.content.strip(), None


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


def _handle_turn(ctx: Dict[str, Any], user_text: str) -> str:
    """대화 턴 처리. /session/text와 /session/voice 모두 이 함수를 사용합니다."""
    print(f"[_handle_turn] 호출: text='{user_text}', step={ctx.get('step')}, category={ctx.get('category')}")
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

    # 일반 질문 감지 → OpenAI로 답변 (UI 위치 질문은 상위에서 이미 처리)
    if looks_like_general_question(text):
        resp_text, _ = answer_general_question(text)
        return resp_text

    # 1) 먹고가기 / 매장에서
    if step == "dine_type":
        # LLM 파싱 시도, 실패 시 규칙 기반 폴백
        dine = _parse_dine_type_llm(text) or _parse_dine_type(text)
        if dine is None:
            return "포장해서 가져가시나요, 매장에서 드시나요?"
        ctx["dine_type"] = dine
        
        # 선택한 옵션을 한국어로 변환
        dine_name = "포장" if dine == "takeout" else "매장"
        
        ctx["step"] = "menu_item"
        return f"{dine_name}을 선택하셨습니다. 원하시는 메뉴를 말씀해주세요."

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
        
        # 복합 액션 체크 ("치즈케이크 빼고 마카롱 담아줘" 등)
        is_complex_action = any(x in t for x in ["빼", "빼줘", "빼고", "빼고나서"]) and any(x in t for x in ["담아", "담아줘", "담아달라", "추가", "넣어", "넣어줘"])
        
        if is_complex_action:
            # 복합 액션 파싱 (제거 + 추가)
            cart_action = _parse_cart_action_llm(text)
            if cart_action:
                remove_menu = cart_action.get("remove_menu", {})
                add_menu = cart_action.get("add_menu", {})
                
                remove_category = remove_menu.get("category")
                remove_menu_id = remove_menu.get("menu_id")
                remove_menu_name = remove_menu.get("menu_name")
                
                add_category = add_menu.get("category")
                add_menu_id = add_menu.get("menu_id")
                add_menu_name = add_menu.get("menu_name")
                
                response_parts = []
                
                # 제거 처리
                if remove_category and remove_menu_id and remove_menu_name:
                    ctx["remove_from_cart"] = True
                    ctx["remove_menu_category"] = remove_category
                    ctx["remove_menu_id"] = remove_menu_id
                    ctx["remove_menu_name"] = remove_menu_name
                    response_parts.append(f"{remove_menu_name}를 장바구니에서 제거했습니다")
                
                # 추가 처리
                if add_category and add_menu_id and add_menu_name:
                    ctx["add_to_cart"] = True
                    # 추가할 메뉴 정보 저장
                    ctx["category"] = add_category
                    ctx["menu_id"] = add_menu_id
                    ctx["menu_name"] = add_menu_name
                    ctx["temp"] = None
                    ctx["size"] = None
                    ctx["options"] = {
                        "extra_shot": 0,
                        "syrup": False,
                        "decaf": None,
                        "sweetness": None,
                    }
                    
                    # 디저트는 바로 추가 가능
                    if add_category == "dessert":
                        response_parts.append(f"{add_menu_name}를 장바구니에 담았습니다")
                    else:
                        # 커피/차/에이드는 온도/사이즈 선택 필요
                        ctx["step"] = "temp" if add_category in ("coffee", "tea") else "size"
                        return f"{add_menu_name}를 선택하셨어요. " + ("따뜻하게 드실까요, 차갑게 드실까요?" if add_category in ("coffee", "tea") else "사이즈는 작은 사이즈, 중간 사이즈, 큰 사이즈 중에서 선택해 주세요.")
                
                ctx["step"] = "menu_item"
                
                if response_parts:
                    return ". ".join(response_parts) + "."
                else:
                    return "메뉴를 다시 말씀해 주세요."
        
        # 장바구니 제거 의도 LLM 감지 ("티라미수 빼줘", "티라미수 장바구니에서 빼줘" 등)
        # "빼", "빼줘", "제거" 등의 키워드가 있으면 LLM으로 제거 의도 확인
        has_remove_keyword = any(x in t for x in ["빼", "빼줘", "빼달라", "빼달라고", "제거", "제거해줘", "삭제", "삭제해줘", "없애", "없애줘"])
        is_remove_from_cart_intent = False
        remove_menu_info = None
        
        if has_remove_keyword:
            # LLM으로 복합 액션 파싱 시도 (제거 의도 확인)
            cart_action = _parse_cart_action_llm(text)
            if cart_action:
                remove_menu = cart_action.get("remove_menu", {})
                # 제거할 메뉴가 있고, 추가할 메뉴가 없는 경우 (순수 제거 의도)
                if remove_menu.get("category") and remove_menu.get("menu_id") and remove_menu.get("menu_name"):
                    add_menu = cart_action.get("add_menu", {})
                    if not add_menu.get("category") or not add_menu.get("menu_id"):
                        is_remove_from_cart_intent = True
                        remove_menu_info = remove_menu
        
        # LLM 감지 실패 시, 규칙 기반 폴백 (장바구니/카트 키워드 필수)
        if not is_remove_from_cart_intent:
            is_remove_from_cart_intent = has_remove_keyword and any(x in t for x in ["장바구니", "카트"])
        
        if is_remove_from_cart_intent:
            # LLM으로 파싱된 정보 사용 또는 메뉴 파싱
            if remove_menu_info:
                parsed_category = remove_menu_info.get("category")
                menu_id = remove_menu_info.get("menu_id")
                menu_name = remove_menu_info.get("menu_name")
            else:
                # 규칙 기반 감지인 경우 메뉴 파싱
                parsed = _parse_menu_item_llm(text, category) or _parse_menu_item(category, text)
                if not parsed:
                    return "어떤 메뉴를 장바구니에서 빼드릴까요? 메뉴 이름을 말씀해 주세요."
                parsed_category, menu_id, menu_name = parsed
            
            # 장바구니에서 제거 플래그 설정
            ctx["remove_from_cart"] = True
            ctx["remove_menu_category"] = parsed_category
            ctx["remove_menu_id"] = menu_id
            ctx["remove_menu_name"] = menu_name
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
            ctx["step"] = "menu_item"
            
            # target_element_id 생성 및 context에 저장
            target_element_id = _menu_id_to_target_element_id(menu_id)
            ctx["target_element_id"] = target_element_id
            
            # 응답 텍스트 생성
            resp_text = f"{menu_name}를 장바구니에서 제거하겠습니다."
            
            return resp_text
        
        # LLM 파싱 시도, 실패 시 규칙 기반 폴백
        parsed = _parse_menu_item_llm(text, category) or _parse_menu_item(category, text)
        if not parsed:
            print(f"[메뉴 파싱 실패] step={step}, category={category}, text='{text}'")
            return "죄송해요, 잘 못 들었어요. 다시 한 번 메뉴를 말씀해 주세요."
        parsed_category, menu_id, menu_name = parsed
        print(f"[메뉴 파싱 성공] category={parsed_category}, menu_id={menu_id}, menu_name={menu_name}")
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
        
        # 메뉴 선택과 함께 장바구니 추가 의도가 있는지 체크 ("담아줘", "담아달라" 등)
        t = text.replace(" ", "").lower()
        is_add_to_cart_intent = any(x in t for x in ["담아", "담아줘", "담아달라", "담아달래", "담아달라고", "담아주", "추가", "넣어", "넣어줘"])
        
        if category in ("coffee", "tea"):
            ctx["step"] = "temp"
            return f"{menu_name}를 선택하셨어요. 따뜻하게 드실까요, 차갑게 드실까요?"
        if category == "ade":
            ctx["step"] = "size"
            return f"{menu_name}를 선택하셨어요. 사이즈는 작은 사이즈, 중간 사이즈, 큰 사이즈 중에서 선택해 주세요."
        if category == "dessert":
            # 디저트는 온도/사이즈 선택이 없으므로, "담아줘" 같은 의도가 있으면 바로 장바구니에 추가
            if is_add_to_cart_intent:
                ctx["add_to_cart"] = True
                ctx["step"] = "menu_item"
                return _cart_added_sentence(ctx)
            else:
                ctx["step"] = "confirm"
                return _order_summary_sentence(ctx)

    # 4) 온도 선택
    if step == "temp":
        # 이전 버튼 클릭 체크
        t = text.replace(" ", "").lower()
        is_back = any(x in t for x in ["이전", "뒤로", "취소", "돌아가", "back", "prev"])
        
        if is_back:
            ctx["step"] = "menu_item"
            return "주문을 다시 진행해주세요."
        
        # LLM 파싱 시도, 실패 시 규칙 기반 폴백
        temp = _parse_temp_llm(text) or _parse_temp(text)

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
        
        # LLM 파싱 시도, 실패 시 규칙 기반 폴백
        size = _parse_size_llm(text) or _parse_size(text)
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

    # 6) 옵션 선택
    if step == "options":
        # 이전 버튼 클릭 체크
        t = text.replace(" ", "").lower()
        is_back = any(x in t for x in ["이전", "뒤로", "취소", "돌아가", "back", "prev"])
        
        if is_back:
            ctx["step"] = "size"
            return "사이즈를 다시 선택해주세요."
        
        options = ctx.get("options", {})
        # LLM 파싱 시도, 실패 시 규칙 기반 폴백
        try:
            parsed_options = _parse_options_llm(category, text, options)
            ctx["options"] = parsed_options
        except Exception as e:
            print(f"[options 파싱] LLM 실패, 규칙 기반 사용: {e}")
            ctx["options"] = _parse_options(category, text, options)
        # 옵션 선택 후 메뉴 정보는 유지하고 메뉴판으로 돌아감
        # 메뉴 + 온도 + 사이즈 + 옵션까지 확정되었으므로 장바구니에 추가
        ctx["add_to_cart"] = True
        ctx["step"] = "menu_item"
        return _cart_added_sentence(ctx)

    # 7) 주문 확인
    if step == "confirm":
        # 이전 버튼 클릭 체크
        t = text.replace(" ", "").lower()
        is_back = any(x in t for x in ["이전", "뒤로", "취소", "돌아가", "back", "prev"])
        
        if is_back:
            ctx["step"] = "menu_item"
            return "주문을 계속 진행해주세요."
        
        # 장바구니에 담아줘 인식
        is_add_to_cart = any(x in t for x in ["장바구니", "담아", "담아줘", "담아주", "추가", "넣어", "넣어줘"])
        
        # 결제하기 버튼 클릭 또는 결제 관련 키워드 체크
        is_payment_intent = any(x in t for x in ["결제하기", "결제", "결제할게요", "결제하겠어요", "결제하겠습니다"])
        
        print(f"받은 텍스트: {text}")
        print(f"전처리 후: {t}")
        print(f"is_payment_intent: {is_payment_intent}")
        print(f"is_add_to_cart: {is_add_to_cart}")
        
        yn = _yes_no(text)
        
        # 결제 의도가 명확하면 결제 수단 파싱 시도
        if is_payment_intent:
            # LLM으로 결제 수단 파싱 시도
            pay = _parse_payment_llm(text) or _parse_payment(text)
            
            if pay:
                # 결제 수단이 명확하면 바로 해당 단계로
                ctx["payment_method"] = pay
                if pay == "card":
                    ctx["step"] = "card"
                    return "카드를 삽입해주세요."
                elif pay == "coupon":
                    ctx["step"] = "coupon"
                    return "아래 바코드기에 핸드폰을 대고 인식시켜주세요."
                else:
                    # 그 외 결제 수단은 바로 완료
                    ctx["step"] = "done"
                    spoken_pay = {
                        "pay": "간편결제",
                        "kakaopay": "카카오페이",
                        "samsungpay": "삼성페이",
                        "cash": "현금",
                    }.get(pay, "선택하신 결제 수단")
                    return f"{spoken_pay}로 결제 도와드릴게요. 주문이 완료되었습니다. 감사합니다."
            else:
                # 결제 수단이 불명확하면 payment 단계로
                ctx["step"] = "payment"
                return "결제 수단을 선택해 주세요. 카드결제, 간편결제, 쿠폰 결제 등으로 말씀해 주세요."
        
        # "네", "맞아요", "장바구니에 담아줘" 등의 표현으로 장바구니 추가
        if yn == "yes" or is_add_to_cart:
            ctx["add_to_cart"] = True
            ctx["step"] = "menu_item"
            return _cart_added_sentence(ctx)
        
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

    # 8) 결제 수단
    if step == "payment":
        # 이전 버튼 클릭 체크
        t = text.replace(" ", "").lower()
        is_back = any(x in t for x in ["이전", "뒤로", "취소", "돌아가", "back", "prev"])
        
        if is_back:
            ctx["step"] = "menu_item"
            return "주문을 계속 진행해주세요."
        
        # 결제 수단 관련 UI 도움말 질문 처리
        # "쿠폰 사용하려면 뭐 눌러야해?", "카드 결제 어떻게 해?", "쿠폰 어디 눌러야 해?" 등
        is_payment_help_question = any(keyword in t for keyword in [
            "어디", "어떻게", "뭐", "무엇", "어떤", "방법", "어디에", "어디서", "어떡해", "어떻게해"
        ]) and any(payment_keyword in t for payment_keyword in [
            "쿠폰", "카드", "결제", "현금", "페이", "카카오"
        ])
        
        if is_payment_help_question:
            # 쿠폰 관련 질문
            if "쿠폰" in t:
                return "쿠폰결제를 눌러주세요."
            # 카드 관련 질문
            if "카드" in t:
                return "카드결제를 눌러주세요."
            # 현금 관련 질문
            if "현금" in t:
                return "현금결제를 눌러주세요."
            # 카카오페이 관련 질문
            if "카카오" in t or "페이" in t:
                return "간편결제를 눌러주세요."
            # 일반적인 결제 수단 질문
            return "카드결제, 간편결제, 쿠폰 결제 중에서 선택해주세요."
        
        # LLM 파싱 시도, 실패 시 규칙 기반 폴백
        pay = _parse_payment_llm(text) or _parse_payment(text)
        if pay is None:
            return "결제 수단을 다시 말씀해 주세요. 카드결제, 간편결제, 쿠폰 결제 등으로 말씀해 주세요."
        ctx["payment_method"] = pay
        
        # 카드 결제인 경우 card 단계로
        if pay == "card":
            ctx["step"] = "card"
            return "카드를 삽입해주세요."
        
        # 쿠폰 결제인 경우 coupon 단계로
        if pay == "coupon":
            ctx["step"] = "coupon"
            return "아래 바코드기에 핸드폰을 대고 인식시켜주세요."
        
        # 그 외 결제 수단은 바로 완료
        ctx["step"] = "done"
        spoken_pay = {
            "pay": "간편결제",
            "kakaopay": "카카오페이",
            "samsungpay": "삼성페이",
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
    
    # 10) 쿠폰 인식 및 결제 완료
    if step == "coupon":
        # 쿠폰 인식 완료 확인 (예: "완료", "인식됐어요", "스캔 완료" 등)
        t = text.replace(" ", "").lower()
        is_complete = any(x in t for x in ["완료", "됐", "인식", "스캔", "결제", "다됐"])
        
        if is_complete:
            ctx["step"] = "done"
            return "쿠폰 결제가 완료되었습니다. 주문이 완료되었습니다. 감사합니다."
        return "아래 바코드기에 핸드폰을 대고 인식시켜주세요."

    # 10) 주문 완료 후 새 주문
    if step == "done":
        ctx.update(_new_session_ctx())
        return "안녕하세요. AI음성 키오스크 말로입니다. 주문을 도와드릴게요."

    # 비정상 상태 → 초기화
    ctx.update(_new_session_ctx())
    return "안녕하세요. AI음성 키오스크 말로입니다. 주문을 도와드릴게요."



# ───────────────────────────────────────────────
# FastAPI 엔드포인트들
# ───────────────────────────────────────────────
@app.on_event("startup")
async def warmup():
    """
    서버 시작 시 STT/TTS 모델 및 관련 리소스를 미리 로딩하여
    첫 번째 요청의 응답 속도를 향상시킵니다.
    """
    print("[Startup] 서버 워밍업 시작...")
    
    try:
        # 1. ffmpeg 경로 확인
        if _FFMPEG_PATH:
            print(f"[Startup] ✓ ffmpeg 경로 확인: {_FFMPEG_PATH}")
        else:
            print("[Startup] ⚠ ffmpeg 경로를 찾을 수 없습니다. 오디오 변환이 실패할 수 있습니다.")
        
        # 2. TTS 디렉토리 생성
        os.makedirs(TTS_DIR, exist_ok=True)
        print(f"[Startup] ✓ TTS 캐시 디렉토리 준비: {TTS_DIR}")
        
        # 3. Whisper API 클라이언트 초기화 및 실제 API 호출로 워밍업
        try:
            # 전역 클라이언트 미리 생성 (whisper_client.py의 전역 캐시 사용)
            whisper_client = make_whisper_client()
            print("[Startup] ✓ Whisper API 클라이언트 생성 완료 (전역 캐시에 저장됨)")
            
            # 더미 오디오 파일 생성 (1초 무음 WAV)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                dummy_wav_path = tmp.name
                # pydub로 간단한 무음 오디오 생성 (1초, 16kHz, mono)
                dummy_audio = AudioSegment.silent(duration=1000, frame_rate=16000)
                dummy_audio.export(dummy_wav_path, format="wav")
            
            try:
                # 더미 오디오로 실제 STT API 호출 (첫 호출 지연을 여기서 처리)
                print("[Startup] Whisper API 첫 호출 중... (이 과정이 첫 요청의 지연을 방지합니다)")
                test_result = transcribe_file(dummy_wav_path, language="ko")
                print(f"[Startup] ✓ Whisper STT 워밍업 완료 (결과: '{test_result[:50] if test_result else '(빈 오디오)'}')")
            except Exception as e:
                print(f"[Startup] ⚠ Whisper STT 워밍업 실패: {e}")
                print("[Startup] 첫 요청이 느릴 수 있습니다.")
            finally:
                # 더미 파일 정리
                try:
                    os.remove(dummy_wav_path)
                except OSError:
                    pass
        except Exception as e:
            print(f"[Startup] ⚠ Whisper 클라이언트 초기화 실패: {e}")
            print("[Startup] 첫 요청이 느릴 수 있습니다.")
        
        # 4. TTS 클라이언트 초기화 및 실제 API 호출로 워밍업
        try:
            # 간단한 텍스트로 실제 TTS API 호출 (첫 호출 지연을 여기서 처리)
            print("[Startup] TTS API 첫 호출 중... (이 과정이 첫 요청의 지연을 방지합니다)")
            test_tts_path = synthesize("테스트", out_path="warmup_test.mp3")
            if os.path.exists(test_tts_path):
                print(f"[Startup] ✓ TTS 워밍업 완료 (캐시 파일: {os.path.basename(test_tts_path)})")
                # 테스트 파일은 캐시로 남겨둠 (나중에 재사용 가능)
            else:
                print("[Startup] ⚠ TTS 테스트 파일 생성 실패")
        except Exception as e:
            print(f"[Startup] ⚠ TTS 워밍업 실패: {e}")
            print("[Startup] 첫 요청이 느릴 수 있습니다.")
        
        # 5. OpenAI GPT 클라이언트 확인
        try:
            # 간단한 테스트 호출 (실제 API 호출 없이 클라이언트만 확인)
            if gpt_client:
                print("[Startup] ✓ OpenAI GPT 클라이언트 준비 완료")
        except Exception as e:
            print(f"[Startup] ⚠ OpenAI GPT 클라이언트 확인 실패: {e}")
        
        # 6. 메뉴 설정 로드
        try:
            menu_cfg, opt_cfg = load_configs()
            print(f"[Startup] ✓ 메뉴 설정 로드 완료 (메뉴 {len(menu_cfg)}개)")
        except Exception as e:
            print(f"[Startup] ⚠ 메뉴 설정 로드 실패: {e}")
        
        print("[Startup] 워밍업 완료! 서버가 요청을 받을 준비가 되었습니다.")
        
    except Exception as e:
        print(f"[Startup] ⚠ 워밍업 중 오류 발생: {e}")
        print("[Startup] 서버는 계속 실행되지만 첫 요청이 느릴 수 있습니다.")


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
        response = {
            "stt_text": payload.text,
            "response_text": maybe,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": None,
        }
        # 세션에 최근 응답 저장
        ctx["last_response"] = {**response, "processed_at": _now()}
        return response

    # 턴 수 가드
    guard = _maybe_close_if_too_long(sid, ctx)
    if guard:
        # 세션에 최근 응답 저장
        ctx["last_response"] = {
            "stt_text": payload.text,
            "response_text": guard["response_text"],
            "tts_path": guard.get("tts_path"),
            "tts_url": _make_tts_url(guard.get("tts_path")) if guard.get("tts_path") else None,
            "context": guard.get("context"),
            "backend_payload": guard.get("backend_payload"),
            "target_element_id": None,
            "processed_at": _now(),
        }
        return guard

    text = (payload.text or "").strip()

    # 1) 이전/뒤로 의도 체크 (UI 도움말 체크보다 우선)
    # 각 step에서 이전 단계로 이동하도록 _handle_turn()에서 처리
    t = text.replace(" ", "").lower()
    is_back_intent = any(x in t for x in [
        "이전", "뒤로", "돌아가", "취소", "back", "prev"
    ])
    
    if is_back_intent:
        # _handle_turn()에서 각 step에 맞게 이전 단계로 이동 처리
        resp_text = _handle_turn(ctx, payload.text)
        tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
        SESS_META[sid] = _now()
        response = {
            "stt_text": payload.text,
            "response_text": resp_text,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": None,
        }
        ctx["last_response"] = {**response, "processed_at": _now()}
        return response

    # 2) 결제 의도 체크 (UI 도움말 체크보다 우선)
    # step이 menu_item이면 confirm으로, confirm이면 payment로 이동
    is_payment_intent = any(x in t for x in [
        "결제하기", "결제", "결제할게요", "결제하겠어요", "결제하겠습니다",
        "결제할게", "결제하자", "결제해줘", "결제해주세요"
    ])
    
    if is_payment_intent:
        current_step = ctx.get("step")
        if current_step == "menu_item":
            # 주문 내역이 있는지 확인
            if ctx.get("menu_name") and ctx.get("category"):
                ctx["step"] = "confirm"
                resp_text = "주문내역을 확인하고 결제를 진행해주세요."
            else:
                resp_text = "주문하실 메뉴를 먼저 선택해 주세요."
        elif current_step == "confirm":
            ctx["step"] = "payment"
            resp_text = "결제 수단을 선택해 주세요. 카드결제, 간편결제, 쿠폰 결제 등으로 말씀해 주세요."
        else:
            # 다른 step에서는 일반 처리
            resp_text = _handle_turn(ctx, payload.text)
            tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
            SESS_META[sid] = _now()
            response = {
                "stt_text": payload.text,
                "response_text": resp_text,
                "tts_path": tts_path,
                "tts_url": _make_tts_url(tts_path) or None,
                "context": _ctx_snapshot(ctx),
                "backend_payload": _build_backend_payload(ctx),
                "target_element_id": None,
            }
            ctx["last_response"] = {**response, "processed_at": _now()}
            return response
        
        tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
        SESS_META[sid] = _now()
        response = {
            "stt_text": payload.text,
            "response_text": resp_text,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": None,
        }
        ctx["last_response"] = {**response, "processed_at": _now()}
        return response
    
    # 3) 프론트에서 is_help=True를 보냈거나, UI 도움말로 보이는 발화면 → UI 모드 (일반 질문보다 먼저 체크)
    # 위치 질문("어디", "어딨어")이 있으면 메뉴명이 있어도 UI 도움말로 처리
    # 메뉴명 + 액션("장바구니에 담아줘", "하나 주세요")이 있으면 메뉴 선택으로 처리
    is_ui_help = looks_like_ui_help(text)
    print(f"[DEBUG /session/text] is_ui_help: {is_ui_help}, text: '{text}'")
    is_menu_with_action = False
    
    # UI 도움말이 아니고 menu_item step이면 메뉴 파싱 시도
    if not is_ui_help and ctx.get("step") == "menu_item":
        test_parsed = _parse_menu_item(ctx.get("category"), text)
        if test_parsed:
            is_menu_with_action = True  # 메뉴가 파싱되면 메뉴 선택 의도
            print(f"[DEBUG /session/text] is_menu_with_action: True (메뉴 파싱 성공)")
    
    print(f"[DEBUG /session/text] 최종 조건: is_ui_help={is_ui_help}, is_menu_with_action={is_menu_with_action}, payload.is_help={payload.is_help}")
    
    if payload.is_help or (is_ui_help and not is_menu_with_action):
        print(f"[DEBUG /session/text] classify_ui_target 호출!")
        # LLM이 UI 요소 위치를 판단하고 메뉴 파싱도 함께 처리
        current_step = ctx.get("step")
        ui_info = classify_ui_target(text, current_step)
        resp_text = ui_info.get(
            "answer_text",
            "어느 버튼을 찾으시는지 다시 한번 말씀해 주세요."
        )
        target_element_id = ui_info.get("target_element_id")

        tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
        SESS_META[sid] = _now()

        response = {
            "stt_text": payload.text,
            "response_text": resp_text,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),           # 주문 상태는 유지
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": target_element_id,  # 프론트에서 하이라이트 용도로 사용
        }
        # 세션에 최근 응답 저장
        ctx["last_response"] = {**response, "processed_at": _now()}
        return response

    # 4) 일반 질문/요청 처리 (텍스트 크기 등) - UI 도움말 체크 이후
    if looks_like_general_question(text):
        resp_text, ui_action = answer_general_question(text)
        tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
        SESS_META[sid] = _now()
        response = {
            "stt_text": payload.text,
            "response_text": resp_text,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": None,
            "ui_action": ui_action,  # 텍스트 크기 조절 등 UI 액션
        }
        ctx["last_response"] = {**response, "processed_at": _now()}
        return response

    # 5) 그 외에는 기존 주문/일반 질문 흐름 사용
    print(f"[POST /session/text] 입력: '{payload.text}', step={ctx.get('step')}, category={ctx.get('category')}")
    
    # target_element_id 초기화 (이전 응답의 target_element_id가 남아있을 수 있음)
    ctx["target_element_id"] = None
    
    resp_text = _handle_turn(ctx, payload.text)
    tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
    SESS_META[sid] = _now()

    # context에서 target_element_id 가져오기 (장바구니 제거 등의 경우 설정됨)
    target_element_id = ctx.get("target_element_id")

    response = {
        "stt_text": payload.text,
        "response_text": resp_text,
        "tts_path": tts_path,
        "tts_url": _make_tts_url(tts_path) or None,
        "context": _ctx_snapshot(ctx),
        "backend_payload": _build_backend_payload(ctx),
        "target_element_id": target_element_id,
    }
    # 세션에 최근 응답 저장
    ctx["last_response"] = {**response, "processed_at": _now()}
    # target_element_id 초기화 (다음 요청을 위해)
    ctx["target_element_id"] = None
    return response


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
        response = {
            "stt_text": user_text,
            "response_text": maybe,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": None,
        }
        # 세션에 최근 응답 저장
        ctx["last_response"] = {**response, "processed_at": _now()}
        return response

    # 턴 수 가드
    guard = _maybe_close_if_too_long(sid, ctx)
    if guard:
        # 세션에 최근 응답 저장
        ctx["last_response"] = {
            "stt_text": user_text,
            "response_text": guard["response_text"],
            "tts_path": guard.get("tts_path"),
            "tts_url": _make_tts_url(guard.get("tts_path")) if guard.get("tts_path") else None,
            "context": guard.get("context"),
            "backend_payload": guard.get("backend_payload"),
            "target_element_id": None,
            "processed_at": _now(),
        }
        return guard

    text = (user_text or "").strip()

    # 음성에서도 UI 도움말 발화면 같은 로직 적용
    print(f"[POST /session/voice] STT 결과: '{text}', step={ctx.get('step')}, category={ctx.get('category')}")
    
    # 1) 이전/뒤로 의도 체크 (UI 도움말 체크보다 우선)
    # 각 step에서 이전 단계로 이동하도록 _handle_turn()에서 처리
    t = text.replace(" ", "").lower()
    is_back_intent = any(x in t for x in [
        "이전", "뒤로", "돌아가", "취소", "back", "prev"
    ])
    
    if is_back_intent:
        # _handle_turn()에서 각 step에 맞게 이전 단계로 이동 처리
        resp_text = _handle_turn(ctx, user_text)
        tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
        SESS_META[sid] = _now()
        response = {
            "stt_text": user_text,
            "response_text": resp_text,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": None,
        }
        ctx["last_response"] = {**response, "processed_at": _now()}
        return response

    # 2) 결제 의도 체크 (UI 도움말 체크보다 우선)
    # step이 menu_item이면 confirm으로, confirm이면 payment로 이동
    t = text.replace(" ", "").lower()
    is_payment_intent = any(x in t for x in [
        "결제하기", "결제", "결제할게요", "결제하겠어요", "결제하겠습니다",
        "결제할게", "결제하자", "결제해줘", "결제해주세요"
    ])
    
    if is_payment_intent:
        current_step = ctx.get("step")
        if current_step == "menu_item":
            # 주문 내역이 있는지 확인
            if ctx.get("menu_name") and ctx.get("category"):
                ctx["step"] = "confirm"
                resp_text = "주문내역을 확인하고 결제를 진행해주세요."
            else:
                resp_text = "주문하실 메뉴를 먼저 선택해 주세요."
        elif current_step == "confirm":
            ctx["step"] = "payment"
            resp_text = "결제 수단을 선택해 주세요. 카드결제, 간편결제, 쿠폰 결제 등으로 말씀해 주세요."
        else:
            # 다른 step에서는 일반 처리
            resp_text = _handle_turn(ctx, user_text)
            tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
            SESS_META[sid] = _now()
            response = {
                "stt_text": user_text,
                "response_text": resp_text,
                "tts_path": tts_path,
                "tts_url": _make_tts_url(tts_path) or None,
                "context": _ctx_snapshot(ctx),
                "backend_payload": _build_backend_payload(ctx),
                "target_element_id": None,
            }
            ctx["last_response"] = {**response, "processed_at": _now()}
            return response
        
        tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
        SESS_META[sid] = _now()
        response = {
            "stt_text": user_text,
            "response_text": resp_text,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": None,
        }
        ctx["last_response"] = {**response, "processed_at": _now()}
        return response
    
    # 3) 위치 질문("어디", "어딨어")이 있으면 메뉴명이 있어도 UI 도움말로 처리 (일반 질문보다 먼저 체크)
    # 메뉴명 + 액션("장바구니에 담아줘", "하나 주세요")이 있으면 메뉴 선택으로 처리
    is_ui_help = looks_like_ui_help(text)
    print(f"[DEBUG /session/voice] is_ui_help: {is_ui_help}, text: '{text}'")
    is_menu_with_action = False
    
    # UI 도움말이 아니고 menu_item step이면 메뉴 파싱 시도
    if not is_ui_help and ctx.get("step") == "menu_item":
        test_parsed = _parse_menu_item(ctx.get("category"), text)
        if test_parsed:
            is_menu_with_action = True  # 메뉴가 파싱되면 메뉴 선택 의도
            print(f"[DEBUG /session/voice] is_menu_with_action: True (메뉴 파싱 성공)")
    
    print(f"[DEBUG /session/voice] 최종 조건: is_ui_help={is_ui_help}, is_menu_with_action={is_menu_with_action}")
    
    if is_ui_help and not is_menu_with_action:
        print(f"[DEBUG /session/voice] classify_ui_target 호출!")
        # LLM이 UI 요소 위치를 판단하고 메뉴 파싱도 함께 처리
        current_step = ctx.get("step")
        ui_info = classify_ui_target(text, current_step)
        resp_text = ui_info.get(
            "answer_text",
            "어느 버튼을 찾으시는지 다시 한번 말씀해 주세요."
        )
        target_element_id = ui_info.get("target_element_id")

        tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
        SESS_META[sid] = _now()

        response = {
            "stt_text": user_text,
            "response_text": resp_text,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": target_element_id,
        }
        # 세션에 최근 응답 저장
        ctx["last_response"] = {**response, "processed_at": _now()}
        return response

    # 4) 일반 질문/요청 처리 (텍스트 크기 등) - UI 도움말 체크 이후
    if looks_like_general_question(text):
        resp_text, ui_action = answer_general_question(text)
        tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
        SESS_META[sid] = _now()
        response = {
            "stt_text": user_text,
            "response_text": resp_text,
            "tts_path": tts_path,
            "tts_url": _make_tts_url(tts_path) or None,
            "context": _ctx_snapshot(ctx),
            "backend_payload": _build_backend_payload(ctx),
            "target_element_id": None,
            "ui_action": ui_action,  # 텍스트 크기 조절 등 UI 액션
        }
        ctx["last_response"] = {**response, "processed_at": _now()}
        return response

    # 5) 그 외에는 기존 주문/일반 질문 흐름 사용
    print(f"[POST /session/voice] _handle_turn 호출: text='{user_text}', step={ctx.get('step')}, category={ctx.get('category')}")
    
    # target_element_id 초기화 (이전 응답의 target_element_id가 남아있을 수 있음)
    ctx["target_element_id"] = None
    
    resp_text = _handle_turn(ctx, user_text)
    tts_path = synthesize(resp_text, out_path=f"response_{sid}.mp3")
    SESS_META[sid] = _now()

    # context에서 target_element_id 가져오기 (장바구니 제거 등의 경우 설정됨)
    target_element_id = ctx.get("target_element_id")

    response = {
        "stt_text": user_text,
        "response_text": resp_text,
        "tts_path": tts_path,
        "tts_url": _make_tts_url(tts_path) or None,
        "context": _ctx_snapshot(ctx),
        "backend_payload": _build_backend_payload(ctx),
        "target_element_id": target_element_id,
    }
    # 세션에 최근 응답 저장
    ctx["last_response"] = {**response, "processed_at": _now()}
    # target_element_id 초기화 (다음 요청을 위해)
    ctx["target_element_id"] = None
    return response


@app.get("/session/state")
def session_state(session_id: str):
    """
    세션 상태 조회. 최근 처리 결과도 포함.
    """
    if session_id not in SESSIONS or _expired(SESS_META.get(session_id, 0)):
        raise HTTPException(status_code=404, detail="세션 없음")
    ctx = SESSIONS[session_id]
    SESS_META[session_id] = _now()
    return _ctx_snapshot(ctx)


@app.get("/session/result")
def session_result(session_id: str):
    """
    최근 처리된 결과 조회 (HTTP POST /session/voice 또는 /session/text로 처리된 결과).
    last_response가 있으면 반환하고, 없으면 null 반환.
    """
    if session_id not in SESSIONS or _expired(SESS_META.get(session_id, 0)):
        raise HTTPException(status_code=404, detail="세션 없음")
    ctx = SESSIONS[session_id]
    SESS_META[session_id] = _now()
    
    last_response = ctx.get("last_response")
    if last_response:
        return last_response
    return None


@app.get("/tts/{filename}")
def get_tts_file(filename: str):
    """생성된 TTS mp3를 내려주는 엔드포인트."""
    path = _tts_path_from_name(filename)
    return FileResponse(path, media_type="audio/mpeg", filename=filename)


