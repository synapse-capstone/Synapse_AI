from enum import Enum, auto
class State(Enum):
    BOOT=auto()
    GREETING=auto()
    MODE_SELECT=auto()
    ORDERING=auto()
    CART_REVIEW=auto()
    PAYMENT_SELECT=auto()
    CONFIRM=auto()
    DONE=auto()
