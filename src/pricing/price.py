from __future__ import annotations
import json, os

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_configs():
    base = os.path.dirname(os.path.dirname(__file__))  # src/
    menu_cfg = _load_json(os.path.join(base, "data", "menu_config.json"))
    opt_cfg  = _load_json(os.path.join(base, "data", "option_config.json"))
    return menu_cfg, opt_cfg

def calc_item_price(item: dict, menu_cfg: dict, opt_cfg: dict) -> int:
    name = item.get("menu")
    size = item.get("size")  # s/m/l
    qty  = int(item.get("qty", 1)) or 1
    if name not in menu_cfg: return 0
    info = menu_cfg[name]
    total = info["base"] + info["size_price"].get(size, 0)

    # 옵션 가산
    if item.get("caffeine") == "decaf":
        total += opt_cfg["caffeine"]["decaf"]
    if int(item.get("extra_shot", 0)) > 0:
        total += int(item["extra_shot"]) * int(opt_cfg["shot"])
    syrup = item.get("syrup")
    if syrup:
        total += opt_cfg["syrup"].get(syrup, 0)
    if item.get("whip") == "추가":
        total += opt_cfg["whip"]

    return total * qty

def calc_cart_total(cart: list[dict], menu_cfg: dict, opt_cfg: dict) -> int:
    return sum(calc_item_price(it, menu_cfg, opt_cfg) for it in cart)
