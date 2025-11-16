Voice Kiosk AI (STT → NLP → TTS)

음성 인식 키오스크의 AI 파트 서버입니다.
사용자의 음성을 받아서:

STT (OpenAI Whisper)로 음성 → 텍스트

규칙 기반 Dialogue Manager(NLP)로 대화 흐름 + 주문 정보 추출

Google Cloud TTS로 텍스트 → 음성

을 수행하고,
프론트엔드에는 응답 텍스트 + 음성 URL,
백엔드에는 주문 JSON(backend_payload)을 제공합니다.

✨ 기능 요약

• STT (Speech-to-Text)

OpenAI Whisper API 사용

/session/voice 에서 음성 파일 업로드 → 텍스트 추출

• 규칙 기반 NLP / Dialogue Manager

간단한 규칙 기반으로 의도/슬롯을 추출

아래 순서로 대화 진행

먹고가기 / 들고가기

메뉴 종류 (커피/차/음료/간식)

온도 (핫/아이스)

사이즈 (톨/그란데/벤티)

옵션 (디카페인, 시럽, 샷, 휘핑 등)

주문 확인

결제수단 선택

• TTS

Google Cloud Text-to-Speech

mp3로 합성 후 /tts/{filename} 로 스트리밍

캐싱으로 중복 비용 최소화

• FastAPI 서버

REST 기반 엔드포인트 제공

session_id 로 사용자별 대화 유지

• backend_payload

최종 주문 JSON

백엔드 팀이 원하는 형태로 쉽게 전달 가능

🗂️ 프로젝트 구조 (AI 파트 중심)

src/
├ dialogue/ 대화 상태·프롬프트·매니저
├ nlp/ 슬롯/의도 추출
├ pricing/ 메뉴·옵션 구성
├ server/ FastAPI (app.py)
├ stt/ Whisper API
├ tts/ Google TTS
├ tests/ pytest 테스트
└ docs/ 문서

현재 버전에서는 대화 흐름과 backend_payload 생성이 app.py 안에 구현되어 있음.
필요 시 dialogue/ 모듈로 리팩토링 가능.

⚙️ 사전 준비

가상환경 활성화
source .venv/bin/activate

.env 파일 작성
OPENAI_API_KEY=xxxx
GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/xxx.json
(절대 Git에 올리면 안 됨)

패키지 설치
pip install -r requirements.txt

▶️ 서버 실행

uvicorn src.server.app:app --reload --port 8000

헬스체크:
curl -s http://127.0.0.1:8000/health

curl -s http://127.0.0.1:8000/version

curl -s http://127.0.0.1:8000/config/menu

📡 주요 API
1) POST /session/start

새로운 세션 생성 + 첫 안내 멘트 반환
• session_id
• response_text
• tts_path / tts_url
• context
• backend_payload=null

2) POST /session/text

텍스트 입력 기반 대화
요청
{ "session_id": "...", "text": "포장" }

응답
• stt_text (=입력 텍스트)
• response_text
• tts_url (프론트 재생용)
• context
• backend_payload (주문 JSON)

3) POST /session/voice

음성 파일 업로드(STT → NLP → TTS)
multipart/form-data
(session_id, audio=file)

응답
• stt_text (Whisper 결과)
• response_text
• tts_url
• context
• backend_payload

4) GET /session/state

session_id 로 현재 대화 상태만 확인
(step, dine_type, category, temp, size, 옵션, 결제수단 등)

5) GET /tts/{filename}

생성된 mp3 음성 스트리밍
프론트에서 <audio> 로 그대로 재생 가능

🔁 예시 흐름 (터미널 텍스트 기반 테스트)

BASE=http://127.0.0.1:8000

SESSION=$(curl -s -X POST $BASE/session/start | python -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')

curl -X POST $BASE/session/text … "포장"
curl -X POST $BASE/session/text … "커피"
curl -X POST $BASE/session/text … "아이스로 주세요"
curl -X POST $BASE/session/text … "톨 사이즈요"
curl -X POST $BASE/session/text … "디카페인에 샷 하나 추가"
curl -X POST $BASE/session/text … "네"
curl -X POST $BASE/session/text … "카드로 할게요"

백엔드로 전송되는 backend_payload 예시:

{
"category": "coffee",
"menu_id": "COFFEE_DEFAULT",
"menu_name": "커피",
"temp": "ice",
"size": "tall",
"quantity": 1,
"options": { "caffeine": "decaf", "syrup": false, "whip": false, "extra_shot": 3 },
"dine_type": "takeout",
"payment_method": "card"
}

🧪 테스트

pytest -q

주요 테스트 파일
• test_sanity.py
• test_slots.py
• test_price.py
• test_dialogue_e2e.py
• test_edge_cases.py

🧰 운영 제약 / 세션 관리

• 세션 TTL = 10분
• 최대 20턴 → 이후 자동 종료 안내
• 허용 오디오: wav, mp3, m4a
• TTS 파일 캐싱: .cache_tts/

🚀 배포 안내

• Render / Cloud Run 등 ASGI 환경에서 동작
• 필요한 환경변수:
OPENAI_API_KEY
GOOGLE_APPLICATION_CREDENTIALS
• GCP 서비스 계정 JSON은 Secret 처리 필요

📄 보안 안내

• API 키와 서비스계정 JSON은 절대 Git에 커밋하지 말 것
• .env는 로컬/서버에서만 관리
• 저장소에는 .env.example 정도만 포함