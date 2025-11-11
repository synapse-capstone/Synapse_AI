from src.dialogue.manager import DialogueCtx, next_turn

def test_empty_reprompt():
    ctx = DialogueCtx()
    # 빈 입력(무음/짧은 발화) → 초기 안내 혹은 재프롬프트가 나와야 함
    out = next_turn(ctx, "")
    assert isinstance(out, str) and len(out) > 0

def test_very_long_text_is_handled():
    ctx = DialogueCtx()
    long_text = "안녕하세요 " * 500  # 아주 긴 입력
    out = next_turn(ctx, long_text)
    # 예외 없이 문자열 응답만 오면 통과
    assert isinstance(out, str) and len(out) > 0
