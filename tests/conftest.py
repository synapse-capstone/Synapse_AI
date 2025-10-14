import os, sys
# 프로젝트 루트(= Capstone)를 sys.path 맨 앞에 추가
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

# 매 테스트마다 CART 비우기
@pytest.fixture(autouse=True)
def reset_cart():
    from src.pipeline import pipeline_mock as pm
    pm.CART.clear()
    yield
    pm.CART.clear()
