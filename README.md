# Voice Kiosk AI (STT → NLP → TTS)

음성 인식 카페 키오스크의 AI 파트 백엔드 서버입니다.
사용자의 음성을 받아서:

1. STT(OpenAI Whisper)로 음성 → 텍스트
2. 규칙 기반 Dialogue Manager(NLP)로 대화 흐름 관리 + 주문 정보 추출
3. OpenAI LLM + 규칙 기반 UI 도움말로 화면 버튼 위치 안내
4. Google Cloud TTS로 텍스트 → 음성(mp3)

을 수행하며,

* 프론트엔드에는: response_text, tts_url, target_element_id
* 백엔드(결제 파트)에는: backend_payload(주문 JSON)

을 제공합니다.

---

## 기능 요약

### STT (Speech-to-Text)

* OpenAI Whisper API 사용
* /session/voice 에서 음성 파일 업로드
* 허용 포맷: wav, mp3, m4a
* 내부 모듈: src/stt/whisper_client.py

### 규칙 기반 NLP / Dialogue Manager

규칙 기반 파서로 사용자의 발화를 해석하여 FSM(상태 기계)로 대화 흐름을 관리합니다.

전체 흐름 순서:

1. 먹고가기 / 들고가기
2. 메뉴 종류 선택(커피, 에이드, 차, 디저트)
3. 메뉴 아이템 선택
4. 온도 선택(hot/ice)
5. 사이즈 선택(tall/grande/venti 등)
6. 옵션 선택(커피 옵션, 에이드 당도 등)
7. 주문 확인
8. 결제수단 선택
9. 주문 완료

모든 상태는 서버 세션(context)에 저장되며, 응답 JSON의 context 필드로 프론트에 전달됩니다.

### UI 도움말 모드

“결제 버튼 어디 있어?”, “장바구니는 어디 있어?” 같은 발화를 처리합니다.

작동 방식:

1. 사용자의 발화가 버튼/위치 질문인지 자동 판별하거나
   또는 프론트에서 is_help: true 로 지정
2. LLM이 찾고 있는 버튼을 해석하여 target_element_id 선택
3. 노인 친화 안내 문장을 생성
4. 서버는 response_text + target_element_id 를 함께 응답

프론트엔드는 target_element_id 를 이용해 해당 UI 요소를 하이라이팅합니다.

지원되는 target_element_id 목록:

* menu_home_button
* menu_pay_button
* menu_cart_area
* temp_prev_button
* temp_next_button
* size_prev_button
* size_next_button
* option_prev_button
* option_next_button
* payment_prev_button
* payment_pay_button
* qr_cancel_button
* qr_send_button

### 일반 안내 질문 모드

“현금 돼?”, “어떻게 해?”, “메뉴 추천해줘?” 같은 질문은 주문 흐름이나 UI가 아니므로
LLM을 통해 짧은 1~2문장 안내 답변을 생성합니다.

target_element_id 는 null로 반환되며 주문 상태는 유지됩니다.

### TTS (Text-to-Speech)

* Google Cloud TTS 사용
* mp3 생성 후 .cache_tts/ 디렉토리에 저장
* /tts/{filename} 엔드포인트로 스트리밍 가능
* 캐싱으로 중복 비용 절감

---

## FastAPI 서버 구조

* 엔트리포인트: src/server/app.py
* 세션/상태 관리
* STT → NLP → TTS 파이프라인 실행
* UI 도움말 / 일반 질문 / 주문 흐름 통합 처리

응답 JSON에는 늘 아래 구조가 포함됩니다:

* stt_text
* response_text
* tts_path
* tts_url
* context
* backend_payload
* target_element_id

---

## backend_payload 구조

백엔드/결제 파트에서 사용하는 주문 JSON입니다.

예시:
{
"category": "coffee",
"menu_id": "COFFEE_AMERICANO",
"menu_name": "아메리카노",
"temp": "ice",
"size": "tall",
"quantity": 1,
"base_price": null,
"options": {
"extra_shot": 2,
"syrup": true,
"decaf": true,
"sweetness": null
},
"dine_type": "takeout",
"payment_method": "card"
}

---

## 프로젝트 구조(AI 파트)

src/

* server/app.py (FastAPI)
* stt/whisper_client.py
* tts/tts_client.py
* pricing/price.py
* (기타 모듈들)


• 세션 TTL = 10분
• 최대 20턴 → 이후 자동 종료 안내
• 허용 오디오: wav, mp3, m4a, 3gp (서버에서 자동으로 WAV로 변환)
• ffmpeg 설치 필요 (PATH 추가 또는 환경변수 FFMPEG_BINARY로 경로 지정)
• TTS 파일 캐싱: .cache_tts/
현재 버전에서는 모든 대화 흐름과 UI 도움말 로직이 app.py 안에 구현되어 있습니다.

---

## 사전 준비

### 1) 가상환경

source .venv/bin/activate

### 2) 환경변수

프로젝트 루트에 .env 작성:
OPENAI_API_KEY=your_api_key
GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service-account.json
BASE_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)

### 3) 패키지 설치

pip install -r requirements.txt

---

## 서버 실행

uvicorn src.server.app:app --reload --port 8000

브라우저 확인:

* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* GET /health
* GET /version
* GET /config/menu

---

## 주요 API

### POST /session/start

새 세션 생성 및 첫 질문 반환.

### POST /session/text

텍스트 입력 기반 흐름 처리.

요청:
{
"session_id": "...",
"text": "포장"
}

응답 필드:
stt_text
response_text
tts_url
context
backend_payload
target_element_id(null 또는 UI ID)

### POST /session/voice

음성 파일 업로드로 STT → NLP → TTS 실행.

### GET /session/state

현재 세션의 대화 상태만 조회.

### GET /tts/{filename}

mp3 스트리밍. 프론트에서 그대로 재생 가능.

---

## 세션 관리 및 제약

* 세션 TTL: 10분
* 최대 20턴 초과 시 자동 초기화
* 허용 오디오: wav, mp3, m4a
* TTS 캐시는 .cache_tts/ 에 저장됨

---

## 보안 안내

* OPENAI_API_KEY, GOOGLE_APPLICATION_CREDENTIALS 등 민감한 키는 Git에 절대 올리지 말 것
* .env 는 로컬/배포 환경에서만 관리
* 저장소에는 .env.example만 포함 권장

---

## 배포 안내

* Render, Cloud Run 등 ASGI 환경에서 uvicorn 기반으로 실행
* 필수 환경 변수:

  * OPENAI_API_KEY
  * GOOGLE_APPLICATION_CREDENTIALS
  * BASE_URL
* GCP 서비스 계정 JSON은 Secret으로 안전하게 저장해야 함

---