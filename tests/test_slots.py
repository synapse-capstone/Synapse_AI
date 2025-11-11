from src.nlp.slots import extract_slots
def test_slots_basic():
    s = extract_slots("아이스 아메리카노 라지 하나 바닐라 시럽 샷 추가")
    assert s["menu"]=="아메리카노" and s["temp"]=="ice" and s["size"]=="l"
    assert s.get("syrup") in ("바닐라","바닐라시럽")
    assert s.get("extra_shot",0)>=1
