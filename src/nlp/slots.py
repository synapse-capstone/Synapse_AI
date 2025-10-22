
import re

def extract_slots(text: str) -> dict:
    t = text.lower()
    out = {}

    # 온도
    if any(k in t for k in ["뜨거", "핫", "따뜻"]): out["temp"]="hot"
    if any(k in t for k in ["아이스","차갑","시원"]): out["temp"]="ice"

    # 사이즈
    if re.search(r"(라지|large| l\b)", t): out["size"]="l"
    elif re.search(r"(미디엄|레귤러|regular| m\b)", t): out["size"]="m"
    elif re.search(r"(스몰|small| s\b)", t): out["size"]="s"

    # 수량
    m = re.search(r"(\d+)\s*(잔|개)", t)
    if m: out["qty"] = int(m.group(1))

    # 옵션
    if "디카페" in t: out["caffeine"]="decaf"
    if "샷" in t and "추가" in t:
        out["extra_shot"] = 1
    if "바닐라" in t: out["syrup"]="바닐라"
    if "휘핑" in t and "빼" not in t: out["whip"]="추가"

    # 메뉴(아주 기초)
    if "아메리카노" in t or "아메" in t: out["menu"]="아메리카노"
    if "라떼" in t and "바닐라" not in t: out["menu"]="라떼"
    if "바닐라라떼" in t or ("바닐라" in t and "라떼" in t): out["menu"]="바닐라라떼"

    # 포장/매장, 모드
    if "포장" in t: out["dine_type"]="takeout"
    if "매장" in t or "먹고" in t: out["dine_type"]="dinein"
    if "음성" in t: out["mode"]="voice"
    if "터치" in t or "화면" in t: out["mode"]="touch"

    # 결제 수단
    if "카드" in t: out["payment"]="card"
    elif "현금" in t: out["payment"]="cash"
    elif "앱결제" in t or "모바일" in t or "페이" in t: out["payment"]="app"

    return out
