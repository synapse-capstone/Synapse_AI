# tests/test_dialogue_e2e.py
from src.dialogue.manager import DialogueCtx, next_turn

def test_e2e_order_flow():
    ctx = DialogueCtx()
    # 인사
    assert "포장" in next_turn(ctx, "")   # GREETING

    # 포장/모드 선택
    next_turn(ctx, "포장")
    next_turn(ctx, "음성")

    # 주문 발화
    out = next_turn(ctx, "아이스 아메리카노 라지 한 잔 샷 추가 바닐라")
    # 주문 반영 여부는 문구가 버전에 따라 다를 수 있으니 유연하게 체크
    assert ("담았어요" in out) or ("계속 주문" in out) or ("담았습니다" in out) or ("아메리카노" in out)

    # 결제 단계 진입
    out = next_turn(ctx, "결제")
    assert ("결제" in out) or ("수단" in out) or ("무엇으로" in out)

    # 결제 수단 선택 → 금액 안내 & 확인 질문
    out = next_turn(ctx, "카드")
    assert ("금액" in out) or ("결제 진행" in out) or ("진행할까요" in out)

    # 결제 최종 확인
    out = next_turn(ctx, "네")
    assert ("대기번호" in out) or ("완료" in out) or ("주문이 완료" in out)
