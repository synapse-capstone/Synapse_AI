from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
from src.dialogue.state import State
from src.dialogue import prompts as P
from src.nlp.slots import extract_slots
from src.nlp.intent_parser import get_intent
from src.pricing.price import load_configs, calc_cart_total

@dataclass
class DialogueCtx:
    state: State = State.BOOT
    slots: Dict[str, Any] = field(default_factory=lambda: {
        "dine_type": None, "mode": None, "menu": None,
        "temp": None, "size": None, "caffeine": None,
        "syrup": None, "whip": None, "extra_shot": 0, "qty": 1
    })
    cart: List[Dict[str, Any]] = field(default_factory=list)
    payment: str | None = None

def _is_item_ready(slots: Dict[str, Any]) -> bool:
    return bool(slots.get("menu") and slots.get("temp") and slots.get("size"))

def _reset_item(slots: Dict[str, Any]):
    slots.update({"menu":None,"temp":None,"size":None,
                  "caffeine":None,"syrup":None,"whip":None,"extra_shot":0,"qty":1})

def _cart_text(cart: List[Dict[str, Any]]) -> str:
    if not cart: return "담긴 내역이 없습니다."
    parts=[]
    for it in cart:
        parts.append(f'{it.get("qty",1)}개 {it.get("temp","")}/{it.get("size","")} {it.get("menu","")}')
    return " , ".join(parts)

def next_turn(ctx: DialogueCtx, user_text: str) -> str:
    # BOOT -> GREETING 자동 전이
    if ctx.state == State.BOOT:
        ctx.state = State.GREETING
        return P.GREETING

    intent = get_intent(user_text)
    slots = extract_slots(user_text)

    # 글로벌 종료
    if intent == "end":
        ctx.state = State.DONE
        return P.DONE_FMT.format(num=23)

    # 상태별 처리
    if ctx.state == State.GREETING:
        if "dine_type" in slots:
            ctx.slots["dine_type"] = slots["dine_type"]
            ctx.state = State.MODE_SELECT
            return P.ASK_MODE
        return P.GREETING

    if ctx.state == State.MODE_SELECT:
        if "mode" in slots:
            ctx.slots["mode"] = slots["mode"]
            if slots["mode"] == "touch":
                ctx.state = State.DONE
                return "터치로 계속 진행해주세요."
            ctx.state = State.ORDERING
            return P.ASK_MENU
        return P.ASK_MODE

    if ctx.state == State.ORDERING:
        # 주문/수정 의도 아니어도 슬롯이 들어오면 반영
        for k,v in slots.items():
            if k in ctx.slots and v is not None:
                ctx.slots[k] = v

        if not ctx.slots["menu"]:
            return P.ASK_MENU
        if not ctx.slots["temp"]:
            return P.ASK_TEMP
        if not ctx.slots["size"]:
            return P.ASK_SIZE

        # 아이템 완성 → 카트 담기
        if _is_item_ready(ctx.slots):
            item = {k: ctx.slots.get(k) for k in ["menu","temp","size","caffeine","syrup","whip","extra_shot","qty"]}
            ctx.cart.append(item)
            _reset_item(ctx.slots)
            ctx.state = State.CART_REVIEW
            return f"{_cart_text(ctx.cart)}. " + P.CART_Q

        return P.ASK_OPTIONS

    if ctx.state == State.CART_REVIEW:
        if intent == "order":
            ctx.state = State.ORDERING
            return P.ASK_MENU
        if intent == "pay":
            if not ctx.cart:
                ctx.state = State.ORDERING
                return P.EMPTY_CART
            ctx.state = State.PAYMENT_SELECT
            return P.ASK_PAYMENT
        # 장바구니 읽어주기
        return f"{_cart_text(ctx.cart)}. " + P.CART_Q

    if ctx.state == State.PAYMENT_SELECT:
        # 결제수단 추출
        if "payment" in slots and slots["payment"]:
            ctx.payment = slots["payment"]
        # 결제수단이 정해졌거나, 사용자가 결제 키워드를 말하면 합계 안내
        if ctx.payment or intent == "pay" or any(k in user_text for k in ["카드","현금","앱"]):
            menu_cfg, opt_cfg = load_configs()
            amount = calc_cart_total(ctx.cart, menu_cfg, opt_cfg)
            ctx.state = State.CONFIRM
            return P.CONFIRM_FMT.format(amount=amount)
        else:
            return P.ASK_PAYMENT


    if ctx.state == State.CONFIRM:
        if intent == "pay" or "네" in user_text or "진행" in user_text:
            ctx.state = State.DONE
            return P.DONE_FMT.format(num=23)
        if "아니" in user_text or "취소" in user_text:
            ctx.state = State.CART_REVIEW
            return f"{_cart_text(ctx.cart)}. " + P.CART_Q
        return "결제를 진행할까요? 네/아니오로 말씀해 주세요."

    if ctx.state == State.DONE:
        return "이용해 주셔서 감사합니다."

    return P.REPROMPT
