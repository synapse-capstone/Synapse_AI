Voice Kiosk AI (STT → NLP → TTS)

음성 인식 키오스크의 AI 파트를 담당하는 서버입니다.
OpenAI Whisper(STT) + 규칙기반 NLP + Google Cloud TTS로 음성 대화 주문을 처리합니다.

✨ 기능 요약

STT: 음성 → 텍스트 (OpenAI Whisper API)

NLP: 의도/슬롯 추출 (포장/매장, 음성/터치, 메뉴·온도·사이즈·옵션, 결제수단)

Dialogue Manager: 상태 전이 기반 대화 흐름 (주문 → 검토 → 결제)

TTS: 텍스트 → 음성 (Google Cloud Text-to-Speech, 캐싱 포함)

FastAPI 서버: REST 엔드포인트 제공

🗂️ 프로젝트 구조

src/
├── dialogue/ : 대화 상태/프롬프트/매니저
├── nlp/ : intent 및 slot 추출
├── pricing/ : 메뉴/옵션 구성 및 가격 계산
├── server/ : FastAPI (app.py)
├── stt/ : Whisper 클라이언트
├── tts/ : Google TTS 클라이언트
├── tests/ : 자동 테스트
└── docs/ : 문서/데이터 (사람이 읽기 쉬운 사본)

⚙️ 사전 준비

Python 가상환경 활성화
source .venv/bin/activate

환경변수 파일 .env 작성 (예시는 .env.example 참고)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_PROJECT=
GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service-account.json

의존성 설치
pip install -r requirements.txt

▶️ 로컬 실행 (Quick Start)

uvicorn src.server.app:app --reload --port 8000

확인 명령어:
curl -s http://127.0.0.1:8000/health
 | python -m json.tool
curl -s http://127.0.0.1:8000/version
 | python -m json.tool
curl -s http://127.0.0.1:8000/config/menu
 | python -m json.tool

📡 주요 API

세션 시작 (POST /session/start)
Response: { "session_id", "response_text", "tts_path" }

텍스트 대화 (POST /session/text)
Body: { "session_id": "...", "text": "포장" }
Response: { "response_text", "tts_path" }

음성 대화 (POST /session/voice)
Form fields: session_id, audio(file)
Response: { "stt_text", "response_text", "tts_path" }

세션 상태 조회 (GET /session/state?session_id=...)
Response: { "state", "slots", "cart", "payment" }

헬스체크 & 버전
GET /health → { "ok": true }
GET /version → { "version": "1.0.0", "stt": "...", "tts": "..." }
GET /config/menu → 메뉴/옵션/가격 정보

🔁 주문 흐름 예시

BASE=http://127.0.0.1:8000

SESSION=$(curl -s -X POST $BASE/session/start | python -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')

curl -s -X POST $BASE/session/text -H "Content-Type: application/json" -d '{"session_id":"'"$SESSION"'","text":"포장"}' | python -m json.tool
curl -s -X POST $BASE/session/text -H "Content-Type: application/json" -d '{"session_id":"'"$SESSION"'","text":"음성"}' | python -m json.tool
curl -s -X POST $BASE/session/text -H "Content-Type: application/json" -d '{"session_id":"'"$SESSION"'","text":"아이스 아메리카노 라지 한 잔 샷 추가 바닐라"}' | python -m json.tool
curl -s -X POST $BASE/session/text -H "Content-Type: application/json" -d '{"session_id":"'"$SESSION"'","text":"결제"}' | python -m json.tool
curl -s -X POST $BASE/session/text -H "Content-Type: application/json" -d '{"session_id":"'"$SESSION"'","text":"카드"}' | python -m json.tool
curl -s -X POST $BASE/session/text -H "Content-Type: application/json" -d '{"session_id":"'"$SESSION"'","text":"네"}' | python -m json.tool

🧪 테스트

pytest -q
(6 passed, warnings는 무시 가능)

테스트 구성

test_sanity.py : 기본 임포트

test_slots.py : 슬롯 추출 규칙

test_price.py : 가격 계산

test_dialogue_e2e.py : 대화 흐름 (주문 → 결제 완료)

test_edge_cases.py : 무음·장문 등 엣지 케이스

🧰 운영 가드

세션 TTL: 10분

턴 수 제한: 20턴

허용 오디오 형식: .wav / .mp3 / .m4a

TTS 캐싱으로 중복비용 절감

🚀 배포 참고

Render 또는 Cloud Run으로 상시 서비스 가능

OPENAI_API_KEY, GOOGLE_APPLICATION_CREDENTIALS 환경변수 필요

GCP 서비스 계정 JSON은 Secret File로 연결

📄 보안 주의사항

API 키와 GCP JSON은 절대 커밋 금지

.env.example만 저장소에 포함하고, .env는 로컬/서버에서 관리