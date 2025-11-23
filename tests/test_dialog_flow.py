from fastapi.testclient import TestClient
from src.server.app import app

client = TestClient(app)


def test_order_coffee_full_flow():
    # 1) 세션 시작
    res = client.post("/session/start")
    assert res.status_code == 200
    sid = res.json()["session_id"]

    # 2) 포장
    res = client.post("/session/text", json={"session_id": sid, "text": "포장"})
    body = res.json()
    assert body["context"]["step"] == "menu_category"
    assert body["context"]["dine_type"] == "takeout"

    # 3) 카테고리: 커피
    res = client.post("/session/text", json={"session_id": sid, "text": "커피"})
    body = res.json()
    assert body["context"]["step"] == "menu_item"
    assert body["context"]["category"] == "coffee"

    # 4) 메뉴: 아메리카노
    res = client.post("/session/text", json={"session_id": sid, "text": "아메리카노"})
    body = res.json()
    assert body["context"]["menu_id"] == "COFFEE_AMERICANO"
    assert body["context"]["menu_name"] == "아메리카노"
    assert body["context"]["step"] == "temp"

    # 5) 온도: 아이스
    res = client.post("/session/text", json={"session_id": sid, "text": "아이스로 주세요"})
    body = res.json()
    assert body["context"]["temp"] == "ice"
    assert body["context"]["step"] == "size"

    # 6) 사이즈: 톨
    res = client.post("/session/text", json={"session_id": sid, "text": "톨 사이즈요"})
    body = res.json()
    assert body["context"]["size"] == "tall"
    assert body["context"]["step"] == "options"

    # 7) 옵션: 디카페인 + 샷 하나 추가
    res = client.post(
        "/session/text",
        json={"session_id": sid, "text": "디카페인에 샷 하나 추가해 주세요"},
    )
    body = res.json()
    assert body["context"]["options"]["decaf"] is True
    assert body["context"]["options"]["extra_shot"] >= 1
    assert body["context"]["step"] == "confirm"
    assert "주문하실 건가요" in body["response_text"]

    # 8) 주문 확인: 네
    res = client.post("/session/text", json={"session_id": sid, "text": "네"})
    body = res.json()
    assert body["context"]["step"] == "payment"

    # 9) 결제: 카드
    res = client.post("/session/text", json={"session_id": sid, "text": "카드로 할게요"})
    body = res.json()
    assert body["context"]["payment_method"] == "card"
    assert body["context"]["step"] == "done"

    # backend_payload 최종 점검
    payload = body["backend_payload"]
    assert payload["category"] == "coffee"
    assert payload["menu_id"] == "COFFEE_AMERICANO"
    assert payload["temp"] == "ice"
    assert payload["size"] == "tall"
    assert payload["options"]["decaf"] is True
    assert payload["payment_method"] == "card"
