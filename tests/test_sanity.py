# src/tests/test_sanity.py
def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_version(client):
    res = client.get("/version")
    assert res.status_code == 200
    assert "version" in res.json()
