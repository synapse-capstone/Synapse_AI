from src.server.app import (
    _parse_dine_type,
    _parse_category,
    _parse_temp,
    _parse_size,
    _parse_menu_item,
)


def test_parse_dine_type():
    assert _parse_dine_type("포장") == "takeout"
    assert _parse_dine_type("먹고 갈게요") == "dinein"
    assert _parse_dine_type("매장에서 먹을게요") == "dinein"


def test_parse_category():
    assert _parse_category("커피 마실래요") == "coffee"
    assert _parse_category("레몬에이드 하나 주세요") == "ade"
    assert _parse_category("녹차 주세요") == "tea"
    assert _parse_category("치즈케이크 먹을래요") == "dessert"


def test_parse_temp():
    assert _parse_temp("아이스로 주세요") == "ice"
    assert _parse_temp("차갑게 해 주세요") == "ice"
    assert _parse_temp("뜨겁게") == "hot"
    assert _parse_temp("따뜻하게 해 주세요") == "hot"


def test_parse_size():
    assert _parse_size("톨 사이즈") == "tall"
    assert _parse_size("그란데로 주세요") == "grande"
    assert _parse_size("벤티") == "venti"
    assert _parse_size("스몰 사이즈") in ("small", "tall", None)  # 구현에 따라 허용


def test_parse_menu_item_coffee():
    assert _parse_menu_item("coffee", "아메리카노") == ("COFFEE_AMERICANO", "아메리카노")
    assert _parse_menu_item("coffee", "에스프레소") == ("COFFEE_ESPRESSO", "에스프레소")
    assert _parse_menu_item("coffee", "카페 라떼") == ("COFFEE_LATTE", "카페 라떼")
    assert _parse_menu_item("coffee", "카푸치노") == ("COFFEE_CAPPUCCINO", "카푸치노")


def test_parse_menu_item_ade():
    assert _parse_menu_item("ade", "레몬에이드") == ("ADE_LEMON", "레몬에이드")
    assert _parse_menu_item("ade", "자몽에이드") == ("ADE_GRAPEFRUIT", "자몽에이드")
    assert _parse_menu_item("ade", "청포도 에이드") == ("ADE_GREEN_GRAPE", "청포도 에이드")
    assert _parse_menu_item("ade", "오렌지 에이드") == ("ADE_ORANGE", "오렌지 에이드")


def test_parse_menu_item_tea():
    assert _parse_menu_item("tea", "캐모마일 티") == ("TEA_CHAMOMILE", "캐모마일 티")
    assert _parse_menu_item("tea", "얼그레이 티") == ("TEA_EARL_GREY", "얼그레이 티")
    assert _parse_menu_item("tea", "유자차") == ("TEA_YUJA", "유자차")
    assert _parse_menu_item("tea", "녹차") == ("TEA_GREEN", "녹차")


def test_parse_menu_item_dessert():
    assert _parse_menu_item("dessert", "치즈케이크") == ("DESSERT_CHEESECAKE", "치즈케이크")
    assert _parse_menu_item("dessert", "티라미수") == ("DESSERT_TIRAMISU", "티라미수")
    assert _parse_menu_item("dessert", "초코 브라우니") == ("DESSERT_BROWNIE", "초코 브라우니")
    assert _parse_menu_item("dessert", "크루아상") == ("DESSERT_CROISSANT", "크루아상")
