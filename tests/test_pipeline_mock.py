from src.pipeline.pipeline_mock import parse_intent, handle, run_once

def test_order_flow():
    msg = "아이스 아메리카노 라지 하나 주세요"
    out = run_once(msg)
    assert "담았어요" in out

def test_pay_without_cart():
    out = run_once("결제할게")
    assert "담긴 내역이 없어요" in out or "먼저 주문" in out
