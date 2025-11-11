# src/pricing/price.py
from __future__ import annotations
import os
import json
from typing import Dict, Tuple, List, Any

# ─────────────────────────────────────────────────────────────
# 설정 파일 경로
# ─────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_MENU_JSON = os.path.abspath(os.path.join(_DATA_DIR, "menu_config.json"))
_OPT_JSON  = os.path.abspath(os.path.join(_DATA_DIR, "option_config.json"))


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"pricing config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    메뉴/옵션 가격 마스터 로드
    returns: (menus, options)
    """
    menus = _read_json(_MENU_JSON)
    opts  = _read_json(_OPT_JSON)
    return menus, opts


# ─────────────────────────────────────────────────────────────
# 가격 계산
# ─────────────────────────────────────────────────────────────
def _price_one_item(item: Dict[str, Any],
                    menus: Dict[str, Any],
                    opts: Dict[str, Any]) -> int:
    """
    단일 아이템 가격 계산
    item 예:
      {
        "menu":"아메리카노","temp":"ice","size":"l",
        "qty":1,"extra_shot":1,"syrup":"바닐라","caffeine":None,"whip":None
      }
    """
    menu_name = item.get("menu")
    if not menu_name or menu_name not in menus:
        # 미정/미지원 메뉴는 0 처리
        return 0

    menu_cfg = menus[menu_name]
    base = int(menu_cfg.get("base", 0))

    # 사이즈 가산
    size = (item.get("size") or "s").lower()
    size_price_map = menu_cfg.get("size_price", {})
    size_add = int(size_price_map.get(size, 0))

    total = base + size_add

    # 샷 추가
    n_shot = int(item.get("extra_shot", 0) or 0)
    shot_unit = int(opts.get("shot", 0))
    total += shot_unit * n_shot

    # 시럽
    syrup = item.get("syrup")
    if syrup:
        syrup_price_map = opts.get("syrup", {})
        total += int(syrup_price_map.get(syrup, 0))

    # 카페인(디카페인 등)
    caffeine = item.get("caffeine")
    if caffeine:
        caf_map = opts.get("caffeine", {})
        total += int(caf_map.get(caffeine, 0))

    # 휘핑
    whip = item.get("whip")
    if whip:
        whip_price = int(opts.get("whip", 0))
        total += whip_price

    # 수량
    qty = int(item.get("qty", 1) or 1)
    if qty < 1:
        qty = 1

    return total * qty


def price_cart(cart: List[Dict[str, Any]]) -> int:
    """
    장바구니 총액 계산 (내부에서 config 로드)
    """
    menus, opts = load_configs()
    total = 0
    for it in (cart or []):
        total += _price_one_item(it, menus, opts)
    return int(total)


def calc_cart_total(cart: List[Dict[str, Any]],
                    menu_cfg: Dict[str, Any] | None = None,
                    opt_cfg: Dict[str, Any] | None = None) -> int:
    """
    manager.py 호환용 시그니처.
    - 외부에서 menu_cfg/opt_cfg를 넘기면 그것을 사용
    - 없으면 내부에서 로드하여 사용
    """
    if menu_cfg is None or opt_cfg is None:
        menu_cfg, opt_cfg = load_configs()
    total = 0
    for it in (cart or []):
        total += _price_one_item(it, menu_cfg, opt_cfg)
    return int(total)
