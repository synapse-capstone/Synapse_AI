import re

from src.nlp.general_qa import answer_general_question, should_route_to_qa

CART = []

MENU = {
    "아메리카노": {"hot": True, "ice": True},
    "라떼": {"hot": True, "ice": True},
    "바닐라 라떼": {"hot": True, "ice": True},
    "카라멜 마키아토": {"hot": True, "ice": True},
}

def parse_intent(text: str):
    t = text.strip()
    # 결제/취소
    if re.search(r"(결제|계산|카드|유지|확인)", t):
        return {"intent":"pay"}
    if re.search(r"(취소|지워)", t):
        return {"intent":"cancel"}

    # 메뉴 + 옵션
    size = "regular"
    if "라지" in t or "large" in t.lower(): size = "large"
    if "스몰" in t or "small" in t.lower(): size = "small"

    temp = "ice" if ("아이스" in t or "ice" in t.lower()) else "hot"

    # 메뉴 탐색 (단순 포함 매칭)
    found = None
    for name in MENU.keys():
        if name in t:
            found = name
            break
    if found:
        return {"intent":"order", "item": found, "temp": temp, "size": size}

    # 장바구니 보기
    if re.search(r"(장바구니|담은|목록)", t):
        return {"intent":"show_cart"}

    return {"intent":"unknown"}

def handle(intent):
    ty = intent["intent"]
    if ty == "order":
        item = intent["item"]
        CART.append({"name": item, "temp": intent["temp"], "size": intent["size"]})
        return f"{intent['size']} { '아이스' if intent['temp']=='ice' else '핫' } {item} 담았어요. 계속 주문하시겠어요, 아니면 결제할까요?"
    if ty == "show_cart":
        if not CART: return "장바구니가 비어 있어요."
        lines = [f"- {c['size']} { '아이스' if c['temp']=='ice' else '핫' } {c['name']}" for c in CART]
        return "장바구니:\n" + "\n".join(lines)
    if ty == "pay":
        if not CART: return "담긴 내역이 없어요. 먼저 주문해 주세요."
        n = len(CART); CART.clear()
        return f"주문이 완료되었습니다. 총 {n}건 처리했습니다. 대기번호는 23번입니다."
    if ty == "cancel":
        CART.clear()
        return "장바구니를 비웠습니다."
    return "죄송해요. 잘 이해하지 못했어요. 다시 말씀해 주시겠어요?"

def run_once(text: str):
    if should_route_to_qa(text):
        return answer_general_question(text)
    intent = parse_intent(text)
    return handle(intent)

if __name__ == "__main__":
    samples = [
        "아이스 아메리카노 라지 하나 주세요",
        "장바구니 보여줘",
        "결제할게",
    ]
    for s in samples:
        print("> 사용자:", s)
        print(run_once(s))
