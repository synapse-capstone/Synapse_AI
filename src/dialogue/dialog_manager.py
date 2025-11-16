from typing import Dict, Any

# 세션별 상태를 임시로 메모리에 저장 (실서비스면 나중에 Redis 같은 걸로 교체 가능)
SESSIONS: Dict[str, Dict[str, Any]] = {}


def get_session_state(session_id: str) -> Dict[str, Any]:
    """세션 상태 가져오기 (없으면 새로 생성)"""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "step": "start",   # start -> choose_mode -> choose_menu ... 이런 식으로 확장 예정
            "mode": None,      # takeout / dinein
        }
    return SESSIONS[session_id]


def handle_user_text(session_id: str, user_text: str) -> str:
    """
    사용자 텍스트를 받아서, 적절한 응답 문장을 반환하는 간단한 대화 로직.
    지금은 '포장 vs 매장'만 처리하고, 나머지는 자리 채우기용.
    """
    state = get_session_state(session_id)
    text = (user_text or "").strip()

    # 1. 처음 시작
    if state["step"] == "start":
        state["step"] = "choose_mode"
        return "포장해서 가져가시나요, 매장에서 드시나요?"

    # 2. 포장 / 매장 선택 단계
    if state["step"] == "choose_mode":
        if "포장" in text:
            state["mode"] = "takeout"
            state["step"] = "choose_menu"
            return "포장을 선택하셨어요. 메뉴를 말씀해 주세요."
        if "매장" in text or "먹고 갈게요" in text or "먹고 갈" in text:
            state["mode"] = "dinein"
            state["step"] = "choose_menu"
            return "매장에서 드시는 걸로 할게요. 메뉴를 말씀해 주세요."

        # 잘 못 알아들었을 때
        return "포장이신가요, 매장에서 드실 건가요? '포장', '매장'이라고 말씀해 주세요."

    # 3. 메뉴 단계 (아직 메뉴 설계 안 했으니 그냥 따라 말하기)
    if state["step"] == "choose_menu":
        if not text:
            return "어떤 메뉴를 드시고 싶으신가요? 예를 들어 '아메리카노 한 잔'처럼 말씀해 주세요."

        # 나중에 여기서 메뉴/옵션/수량 파싱하면 됨
        return f"알겠습니다. 지금은 아직 메뉴 세부 옵션은 준비 중이에요. 방금 말씀하신 내용은: '{text}' 입니다."

    # 혹시 모르는 상태일 때
    return "대화를 다시 시작할게요. '포장' 또는 '매장'이라고 말씀해 주세요."
