# 프론트엔드 연동 가이드

## API 기본 정보

- **Base URL**: `http://127.0.0.1:8000` (개발 환경)
- **API 문서**: `http://127.0.0.1:8000/docs` (Swagger UI)
- **서버 실행**: `uvicorn src.server.app:app --reload`

---

## 주요 API 엔드포인트

### 1. 세션 시작
**POST** `/session/start`

세션을 시작하고 초기 인사 메시지를 받습니다.

**요청:**
```json
{}
```

**응답:**
```json
{
  "session_id": "abc123...",
  "response_text": "안녕하세요. AI음성 키오스크 말로입니다. 주문을 도와드릴게요.",
  "tts_path": ".cache_tts/xxx.mp3",
  "tts_url": "http://127.0.0.1:8000/tts/xxx.mp3",
  "context": {
    "step": "greeting",
    "dine_type": null,
    "category": null,
    "menu_id": null,
    "menu_name": null,
    ...
  },
  "backend_payload": null,
  "target_element_id": null
}
```

**중요 필드:**
- `session_id`: 이후 모든 요청에 포함해야 하는 세션 ID
- `response_text`: 사용자에게 보여줄/말할 응답 텍스트
- `tts_url`: 음성 재생용 URL
- `context`: 현재 주문 상태 (step, dine_type, menu 등)
- `target_element_id`: UI 요소 하이라이트용 (예: "menu_item_coffee_americano")

---

### 2. 텍스트 입력
**POST** `/session/text`

사용자의 텍스트 입력을 처리합니다.

**요청:**
```json
{
  "session_id": "abc123...",
  "text": "아메리카노 선택할게",
  "is_help": false  // 선택사항: 도움말 모드 플래그
}
```

**응답:**
```json
{
  "stt_text": "아메리카노 선택할게",
  "response_text": "아메리카노를 선택하셨어요. 따뜻하게 드실까요, 차갑게 드실까요?",
  "tts_path": ".cache_tts/xxx.mp3",
  "tts_url": "http://127.0.0.1:8000/tts/xxx.mp3",
  "context": {
    "step": "temp",
    "dine_type": "takeout",
    "category": "coffee",
    "menu_id": "COFFEE_AMERICANO",
    "menu_name": "아메리카노",
    "temp": null,
    "size": null,
    ...
  },
  "backend_payload": {
    "category": "coffee",
    "menu_id": "COFFEE_AMERICANO",
    "menu_name": "아메리카노",
    "add_to_cart": false,
    ...
  },
  "target_element_id": null
}
```

**특수 응답 케이스:**

#### UI 도움말 응답 (target_element_id 포함)
```json
{
  "response_text": "아메리카노는 메뉴판 상단 커피 섹션에 있습니다.",
  "target_element_id": "menu_item_coffee_americano",  // 프론트에서 하이라이트
  ...
}
```

#### 장바구니 추가 응답
```json
{
  "response_text": "아메리카노가 장바구니에 담겼습니다...",
  "backend_payload": {
    "category": "coffee",
    "menu_id": "COFFEE_AMERICANO",
    "add_to_cart": true,  // 프론트에서 장바구니에 추가
    ...
  },
  ...
}
```

#### 장바구니 제거 응답
```json
{
  "response_text": "치즈케이크를 장바구니에서 제거했습니다.",
  "backend_payload": {
    "remove_from_cart": true,  // 프론트에서 장바구니에서 제거
    "remove_menu": {
      "category": "dessert",
      "menu_id": "DESSERT_CHEESECAKE",
      "menu_name": "치즈케이크"
    },
    ...
  },
  ...
}
```

#### 복합 액션 응답 (제거 + 추가)
```json
{
  "response_text": "치즈케이크를 장바구니에서 제거했습니다. 마카롱를 장바구니에 담았습니다.",
  "backend_payload": {
    "remove_from_cart": true,
    "remove_menu": {
      "category": "dessert",
      "menu_id": "DESSERT_CHEESECAKE",
      "menu_name": "치즈케이크"
    },
    "add_to_cart": true,
    "category": "dessert",
    "menu_id": "DESSERT_MACARON",
    "menu_name": "마카롱",
    ...
  },
  ...
}
```

---

### 3. 음성 입력
**POST** `/session/voice?session_id={session_id}`

사용자의 음성 파일을 업로드하여 처리합니다.

**요청:**
- `Content-Type`: `multipart/form-data`
- `audio`: 오디오 파일 (`.wav`, `.mp3`, `.m4a`, `.3gp`)
- Query Parameter: `session_id`

**응답:** `/session/text`와 동일한 형식

**예시:**
```javascript
const formData = new FormData();
formData.append('audio', audioFile);

const response = await fetch(
  `http://127.0.0.1:8000/session/voice?session_id=${sessionId}`,
  {
    method: 'POST',
    body: formData
  }
);
```

---

### 4. 상태 확인
**GET** `/session/state?session_id={session_id}`

현재 세션 상태를 확인합니다.

**응답:**
```json
{
  "step": "menu_item",
  "dine_type": "takeout",
  "category": "coffee",
  ...
}
```

---

### 5. TTS 파일 재생
**GET** `/tts/{filename}`

생성된 음성 파일을 재생합니다.

**URL 예시:** `http://127.0.0.1:8000/tts/abc123def456.mp3`

---

### 6. 메뉴 정보 조회
**GET** `/config/menu`

메뉴 구성 정보를 조회합니다.

**응답:**
```json
{
  "menus": { ... },
  "options": { ... }
}
```

---

## 프론트엔드 처리 가이드

### 1. 세션 관리

```javascript
// 세션 시작
const startSession = async () => {
  const response = await fetch('http://127.0.0.1:8000/session/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
  const data = await response.json();
  const sessionId = data.session_id;
  // sessionId를 저장 (localStorage 등)
  return data;
};
```

### 2. 응답 처리

```javascript
const handleResponse = (response) => {
  // 1. 응답 텍스트 표시
  displayResponseText(response.response_text);
  
  // 2. 음성 재생
  if (response.tts_url) {
    playAudio(response.tts_url);
  }
  
  // 3. UI 요소 하이라이트
  if (response.target_element_id) {
    highlightElement(response.target_element_id);
  }
  
  // 4. 장바구니 처리
  if (response.backend_payload) {
    const payload = response.backend_payload;
    
    // 장바구니 추가
    if (payload.add_to_cart) {
      addToCart({
        category: payload.category,
        menu_id: payload.menu_id,
        menu_name: payload.menu_name,
        temp: payload.temp,
        size: payload.size,
        options: payload.options,
        ...
      });
    }
    
    // 장바구니 제거
    if (payload.remove_from_cart) {
      const removeMenu = payload.remove_menu || {
        category: payload.category,
        menu_id: payload.menu_id,
        menu_name: payload.menu_name
      };
      removeFromCart(removeMenu);
    }
  }
  
  // 5. 화면 전환 (context.step 기반)
  const step = response.context?.step;
  switch(step) {
    case 'greeting':
      showGreetingScreen();
      break;
    case 'dine_type':
      showDineTypeScreen();
      break;
    case 'menu_item':
      showMenuScreen();
      break;
    case 'temp':
      showTempScreen();
      break;
    case 'size':
      showSizeScreen();
      break;
    case 'options':
      showOptionsScreen();
      break;
    case 'confirm':
      showConfirmScreen();
      break;
    case 'payment':
      showPaymentScreen();
      break;
    case 'card':
      showCardScreen();
      break;
    case 'coupon':
      showCouponScreen();
      break;
    case 'done':
      showDoneScreen();
      break;
  }
};
```

### 3. target_element_id 하이라이트

```javascript
const highlightElement = (targetElementId) => {
  // 1. 기존 하이라이트 제거
  document.querySelectorAll('.highlighted').forEach(el => {
    el.classList.remove('highlighted');
  });
  
  // 2. 해당 요소 찾기
  const element = document.getElementById(targetElementId) || 
                  document.querySelector(`[data-element-id="${targetElementId}"]`);
  
  // 3. 하이라이트 적용
  if (element) {
    element.classList.add('highlighted');
    element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // 애니메이션 효과 (깜빡임 등)
    element.style.animation = 'blink 1s ease-in-out 3';
  }
};
```

### 4. 장바구니 처리

```javascript
// 장바구니 추가
const addToCart = (item) => {
  // item 구조:
  // {
  //   category: "coffee",
  //   menu_id: "COFFEE_AMERICANO",
  //   menu_name: "아메리카노",
  //   temp: "ice",
  //   size: "grande",
  //   options: { ... },
  //   quantity: 1
  // }
  
  // 프론트엔드 장바구니에 추가 로직
  cart.push(item);
  updateCartUI();
};

// 장바구니 제거
const removeFromCart = (menuInfo) => {
  // menuInfo 구조:
  // {
  //   category: "dessert",
  //   menu_id: "DESSERT_CHEESECAKE",
  //   menu_name: "치즈케이크"
  // }
  
  // 프론트엔드 장바구니에서 제거 로직
  cart = cart.filter(item => 
    !(item.category === menuInfo.category && 
      item.menu_id === menuInfo.menu_id)
  );
  updateCartUI();
};
```

### 5. 복합 액션 처리 (제거 + 추가)

```javascript
if (payload.remove_from_cart && payload.add_to_cart) {
  // 1. 제거할 메뉴 제거
  if (payload.remove_menu) {
    removeFromCart(payload.remove_menu);
  }
  
  // 2. 추가할 메뉴 추가
  addToCart({
    category: payload.category,
    menu_id: payload.menu_id,
    menu_name: payload.menu_name,
    temp: payload.temp,
    size: payload.size,
    options: payload.options,
    ...
  });
}
```

---

## Context Step 값

| Step | 설명 | 화면 전환 |
|------|------|-----------|
| `greeting` | 초기 인사 | 인사 화면 |
| `dine_type` | 포장/매장 선택 | 포장/매장 선택 화면 |
| `menu_item` | 메뉴 선택 | 메뉴 목록 화면 |
| `temp` | 온도 선택 | 온도 선택 화면 |
| `size` | 사이즈 선택 | 사이즈 선택 화면 |
| `options` | 옵션 선택 | 옵션 선택 화면 |
| `confirm` | 주문 확인 | 주문 확인 화면 |
| `payment` | 결제 수단 선택 | 결제 수단 선택 화면 |
| `card` | 카드 결제 | 카드 결제 화면 |
| `coupon` | 쿠폰 결제 | 쿠폰 인식 화면 |
| `done` | 주문 완료 | 완료 화면 |

---

## Target Element ID 목록

### 메뉴 화면
- `menu_home_button`: 홈 버튼
- `menu_pay_button`: 결제하기 버튼
- `menu_cart_area`: 장바구니 영역

### 메뉴 아이템
- `menu_item_coffee_americano`: 아메리카노
- `menu_item_coffee_latte`: 카페 라떼
- `menu_item_dessert_cheesecake`: 치즈케이크
- `menu_item_dessert_macaron`: 마카롱
- ... (전체 목록은 API 응답 참고)

---

## 주요 기능별 처리 예시

### 1. UI 도움말 (위치 질문)
```javascript
// 사용자: "아메리카노 어딨어?"
// 응답:
{
  "response_text": "아메리카노는 메뉴판 상단 커피 섹션에 있습니다.",
  "target_element_id": "menu_item_coffee_americano"
}

// 처리:
if (response.target_element_id) {
  highlightElement(response.target_element_id);
}
```

### 2. 장바구니 추가
```javascript
// 사용자: "티라미수 담아줘"
// 응답:
{
  "response_text": "티라미수가 장바구니에 담겼습니다...",
  "backend_payload": {
    "add_to_cart": true,
    "category": "dessert",
    "menu_id": "DESSERT_TIRAMISU",
    "menu_name": "티라미수",
    ...
  }
}

// 처리:
if (response.backend_payload?.add_to_cart) {
  addToCart(response.backend_payload);
}
```

### 3. 장바구니 제거
```javascript
// 사용자: "치즈케이크 장바구니에서 빼줘"
// 응답:
{
  "response_text": "치즈케이크를 장바구니에서 제거했습니다.",
  "backend_payload": {
    "remove_from_cart": true,
    "remove_menu": {
      "category": "dessert",
      "menu_id": "DESSERT_CHEESECAKE",
      "menu_name": "치즈케이크"
    }
  }
}

// 처리:
if (response.backend_payload?.remove_from_cart) {
  const removeMenu = response.backend_payload.remove_menu || response.backend_payload;
  removeFromCart(removeMenu);
}
```

### 4. 복합 액션 (제거 + 추가)
```javascript
// 사용자: "치즈케이크 빼고 마카롱 담아줘"
// 응답:
{
  "response_text": "치즈케이크를 장바구니에서 제거했습니다. 마카롱를 장바구니에 담았습니다.",
  "backend_payload": {
    "remove_from_cart": true,
    "remove_menu": {
      "category": "dessert",
      "menu_id": "DESSERT_CHEESECAKE",
      "menu_name": "치즈케이크"
    },
    "add_to_cart": true,
    "category": "dessert",
    "menu_id": "DESSERT_MACARON",
    "menu_name": "마카롱",
    ...
  }
}

// 처리:
const payload = response.backend_payload;
if (payload.remove_from_cart && payload.remove_menu) {
  removeFromCart(payload.remove_menu);
}
if (payload.add_to_cart) {
  addToCart(payload);
}
```

---

## 에러 처리

### 세션 만료
세션이 10분간 비활성화되면 자동으로 만료됩니다. 새 세션을 시작하세요.

### API 오류
```javascript
try {
  const response = await fetch(...);
  if (!response.ok) {
    const error = await response.json();
    console.error('API 오류:', error);
    // 사용자에게 오류 메시지 표시
  }
  const data = await response.json();
  handleResponse(data);
} catch (error) {
  console.error('네트워크 오류:', error);
  // 사용자에게 네트워크 오류 메시지 표시
}
```

---

## 개발 팁

1. **세션 ID 저장**: `localStorage` 또는 상태 관리에 세션 ID를 저장
2. **TTS 재생**: `Audio` 객체나 `<audio>` 태그로 재생
3. **하이라이트**: CSS 애니메이션 또는 라이브러리 사용
4. **상태 동기화**: `context.step`을 기반으로 화면 전환
5. **디버깅**: 브라우저 개발자 도구 Network 탭에서 API 호출 확인

---

## API 문서 확인

실제 구현된 API 스펙을 확인하려면:
1. 서버 실행: `uvicorn src.server.app:app --reload`
2. 브라우저에서 `http://127.0.0.1:8000/docs` 접속
3. Swagger UI에서 각 엔드포인트 테스트 및 스키마 확인

