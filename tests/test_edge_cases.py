# src/tests/test_edge_cases.py

def test_empty_input(client):
    res = client.post("/session/start")
    sid = res.json()["session_id"]

    res = client.post("/session/text", json={"session_id": sid, "text": ""})
    assert "잘 못 들었어요" in res.json()["response_text"]


def test_wrong_menu(client):
    res = client.post("/session/start")
    sid = res.json()["session_id"]

    client.post("/session/text", json={"session_id": sid, "text": "포장"})
    client.post("/session/text", json={"session_id": sid, "text": "커피"})

    res = client.post("/session/text", json={"session_id": sid, "text": "이상한메뉴"})
    assert "죄송해요" in res.json()["response_text"]


def test_restart_after_no(client):
    res = client.post("/session/start")
    sid = res.json()["session_id"]

    # 포장 → 커피 → 아메리카노 → temp → size → options → confirm
    client.post("/session/text", json={"session_id": sid, "text": "포장"})
    client.post("/session/text", json={"session_id": sid, "text": "커피"})
    client.post("/session/text", json={"session_id": sid, "text": "아메리카노"})
    client.post("/session/text", json={"session_id": sid, "text": "아이스"})
    client.post("/session/text", json={"session_id": sid, "text": "톨"})
    client.post("/session/text", json={"session_id": sid, "text": "샷 추가"})

    # 주문 확인에서 “아니요”
    res = client.post("/session/text", json={"session_id": sid, "text": "아니요"})
    assert res.json()["context"]["step"] == "menu_category"
