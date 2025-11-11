from src.pricing.price import price_cart
def test_price_cart():
    cart=[{"menu":"아메리카노","temp":"ice","size":"l","qty":1,"extra_shot":1,"syrup":"바닐라"}]
    total = price_cart(cart)
    assert total > 0
