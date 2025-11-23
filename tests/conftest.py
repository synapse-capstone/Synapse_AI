import os
import sys

# 프로젝트 루트(= 이 파일 기준으로 상위 디렉터리)를 PYTHONPATH에 추가
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi.testclient import TestClient
from src.server.app import app


import pytest


@pytest.fixture
def client():
    return TestClient(app)
