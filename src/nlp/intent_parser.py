def get_intent(text: str) -> str:
    text = text.strip().lower()
    if any(k in text for k in ["주문", "주세요", "갖다", "아이스", "라떼"]):
        return "order"
    if any(k in text for k in ["결제", "카드", "현금", "계산"]):
        return "pay"
    if any(k in text for k in ["취소", "다시", "없애"]):
        return "cancel"
    if any(k in text for k in ["끝", "그만", "종료", "안녕"]):
        return "end"
    return "unknown"
