from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")
OPENAI_QA_MODEL = os.getenv("OPENAI_QA_MODEL", "gpt-4o-mini")

QUESTION_KEYWORDS: Iterable[str] = (
    "?", "무엇", "뭐야", "뭔가", "어떤", "어떻게", "왜", "알려줘", "설명", "차이", "정보"
)

GENERAL_TOPICS: Iterable[str] = (
    "커피", "음료", "메뉴", "디저트", "아메리카노", "라떼", "마끼아또", "마키아토",
)


def _has_keyword(text: str, keywords: Iterable[str]) -> bool:
    return any(word in text for word in keywords)


def should_route_to_qa(user_text: str) -> bool:
    """간단한 휴리스틱으로 일반 지식 질문 여부 판별."""
    if not user_text:
        return False
    text = user_text.strip()
    if len(text) < 3:
        return False
    if _has_keyword(text, QUESTION_KEYWORDS) and _has_keyword(text, GENERAL_TOPICS):
        return True
    return text.endswith("야") or text.endswith("죠") or text.endswith("나요")


@lru_cache(maxsize=1)
def _make_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    if OPENAI_PROJECT:
        return OpenAI(
            api_key=OPENAI_API_KEY,
            project=OPENAI_PROJECT,
            default_headers={"OpenAI-Project": OPENAI_PROJECT},
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def answer_general_question(user_text: str) -> str:
    """LLM 호출로 메뉴/일반 지식 질문 답변."""
    try:
        client = _make_client()
    except RuntimeError:
        return "지식 답변 기능을 사용할 수 없어요. 관리자에게 문의해 주세요."

    prompt = user_text.strip()
    if not prompt:
        return "어떤 내용을 설명해 드릴까요?"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_QA_MODEL,
            temperature=0.2,
            max_tokens=400,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 카페 키오스크 도우미야. "
                        "사용자가 일반 지식이나 메뉴 설명을 물으면 친절하고 짧게 한국어로 알려줘."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        answer = resp.choices[0].message.content.strip()
        return answer or "설명할 내용을 다시 알려주시겠어요?"
    except Exception:
        return "지금은 답변을 생성하기 어렵습니다. 잠시 후 다시 시도해 주세요."


