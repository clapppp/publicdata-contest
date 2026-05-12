# Silver Voice Resume API — 상세 명세

**Base URL** (예시): `https://0oof8muhxkpy97-8000.proxy.runpod.net`  
**모델**: `claude-haiku-4-5`  
**인증**: 없음 (서버 내부 ANTHROPIC_API_KEY 사용)

---

## 공통

- 모든 HTTP 요청/응답 Content-Type: `application/json`
- 오류 응답 형식: `{ "detail": "오류 메시지" }`
- `user_id`는 Flutter 앱이 생성·관리하는 임의 문자열 (UUID 권장)

---

## 1. 서버 상태 확인

### `GET /health`

서버 및 Claude API 연결 상태를 확인합니다.

**요청**: 없음

**응답 200**
```json
{
  "status": "ok",
  "model": "claude-haiku-4-5"
}
```

---

## 2. 이력서 API

이력서 구조 (공통):

| 필드 | 타입 | 설명 |
|---|---|---|
| `name` | string | 이름 |
| `age` | int | 나이 (0이면 미입력) |
| `location` | string | 거주지 |
| `career` | string | 경력 |
| `preferred_work_type` | string | 희망 근무형태 |
| `physical_condition` | string | 건강 상태 |

응답에 포함되는 추가 필드:

| 필드 | 타입 | 설명 |
|---|---|---|
| `resume` | object | 위 이력서 구조 |
| `missing_korean` | string[] | 아직 비어있는 항목 이름 목록 (한국어) |

---

### `GET /resume/{user_id}`

저장된 이력서와 빈 항목 목록을 가져옵니다.

**응답 200**
```json
{
  "resume": {
    "name": "홍길동",
    "age": 67,
    "location": "서울 강남구",
    "career": "건설현장 십년",
    "preferred_work_type": "",
    "physical_condition": ""
  },
  "missing_korean": ["희망 근무형태", "건강 상태"]
}
```

**응답 400**: user_id가 올바르지 않을 때

---

### `POST /resume/{user_id}`

이력서를 직접 수정합니다. **보낸 필드만 업데이트**됩니다 (부분 수정).  
음성 챗 없이 프론트에서 직접 이력서를 편집할 때 사용합니다.

**요청 Body** (모든 필드 선택 사항 — 바꿀 것만 보내세요)
```json
{
  "name": "홍길동",
  "age": 67,
  "location": "서울 강남구",
  "career": "건설현장 십년",
  "preferred_work_type": "주 3일 단기",
  "physical_condition": "무릎이 좀 안 좋음"
}
```

**응답 200** — 업데이트된 전체 이력서 반환 (GET /resume 형식과 동일)
```json
{
  "resume": { ... },
  "missing_korean": []
}
```

**응답 400**: user_id가 올바르지 않을 때

---

### `DELETE /resume/{user_id}`

이력서와 대화 이력을 모두 초기화합니다.

**응답 200**
```json
{
  "status": "ok",
  "user_id": "user-abc123"
}
```

---

## 3. 채용 추천 API

### `POST /recommend/{user_id}`

저장된 이력서를 기반으로 채용공고를 추천합니다.  
이력서 미완성 시 추천 없이 안내 메시지를 반환합니다.

**Query Params**

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `top_k` | int | 5 | 추천 개수 |

**요청**: 없음 (Body 불필요)

**응답 200 — 정상 추천**
```json
{
  "recommendations": [
    {
      "job_id": "JOB-001",
      "job_title": "아파트 경비원",
      "company": "○○관리공단",
      "score": 92,
      "reason": "경비 경력과 체력 조건이 잘 맞습니다."
    }
  ]
}
```

**응답 200 — 이력서 미완성**
```json
{
  "recommendations": [],
  "message": "이력서가 아직 미완성입니다. 음성 채팅으로 채워주세요.",
  "missing": ["희망 근무형태", "건강 상태"]
}
```

**응답 200 — 매칭 없음**
```json
{
  "recommendations": [],
  "message": "조건에 맞는 채용공고가 없습니다."
}
```

---

### `POST /recommend`

이력서를 Body에 직접 담아 추천 요청합니다.  
저장된 이력서 없이 즉시 추천이 필요할 때 사용합니다.

**요청 Body**
```json
{
  "resume": {
    "name": "홍길동",
    "age": 67,
    "location": "서울 강남구",
    "career": "건설현장 십년",
    "preferred_work_type": "주 3일",
    "physical_condition": "무릎이 좀 안 좋음"
  },
  "top_k": 5
}
```

**응답 200** — `/recommend/{user_id}` 정상 추천 응답과 동일 형식

---

## 4. 채용공고 재수집

### `POST /refresh`

100세누리 API에서 채용공고를 다시 가져와 RAG DB(ChromaDB)를 업데이트합니다.  
주기적으로 또는 수동으로 호출하세요.

**응답 200**
```json
{
  "status": "ok",
  "before": 120,
  "after": 135,
  "delta": 15
}
```

---

## 5. 음성 챗 WebSocket

### `WS /voice/ws`

음성 → STT → LLM → 이력서 자동 추출까지 처리하는 메인 엔드포인트입니다.  
TTS는 앱(Flutter)의 디바이스 TTS가 `llm_token`을 받아 직접 처리합니다.

**WebSocket URL**: `wss://<host>/voice/ws`

---

### 메시지 흐름

```
[연결]
  서버 → { "type": "ready" }
  서버 → { "type": "greeting", "text": "안녕하세요! ..." }

[이력서 상태 조회 (선택)]
  클라 → { "type": "sync_resume", "user_id": "..." }
  서버 → { "type": "resume_state", "resume": {...}, "missing_korean": [...] }

[발화 1회 처리]
  클라 → { "type": "turn", "user_id": "..." }          (text JSON)
  클라 → <audio binary>                                (AAC/webm/opus 등)

  서버 → { "type": "resume_state", "resume": {...}, "missing_korean": [...] }
  서버 → { "type": "transcript", "text": "STT 결과" }
  서버 → { "type": "llm_token", "text": "토큰1" }      (N회 반복)
  서버 → { "type": "llm_token", "text": "토큰2" }
  ...
  서버 → { "type": "resume_updated", "resume": {...}, "missing_korean": [...] }
  서버 → { "type": "reply_done" }

[오류]
  서버 → { "type": "error", "msg": "오류 내용" }
```

---

### 클라이언트 → 서버 메시지

#### `sync_resume`
현재 이력서 상태를 요청합니다. 연결 직후나 화면 진입 시 호출하세요.

```json
{ "type": "sync_resume", "user_id": "user-abc123" }
```

#### `turn`
새 발화 시작을 알립니다. 이 JSON 직후 오디오 바이너리를 전송합니다.

```json
{ "type": "turn", "user_id": "user-abc123" }
```

이어서 녹음된 오디오를 `binary` 프레임으로 전송합니다.  
포맷: AAC, webm/opus, wav 등 ffmpeg이 디코딩 가능한 포맷.

---

### 서버 → 클라이언트 메시지

| type | 추가 필드 | 설명 |
|---|---|---|
| `ready` | — | 연결 직후 1회. 이 시점부터 메시지 송신 가능 |
| `greeting` | `text` | 첫 안내 문구. TTS로 재생하세요 |
| `resume_state` | `resume`, `missing_korean` | 현재 이력서 스냅샷 |
| `transcript` | `text` | STT 변환 결과 |
| `llm_token` | `text` | LLM 응답 토큰 스트림. 받는 즉시 TTS 큐에 추가하세요 |
| `resume_updated` | `resume`, `missing_korean` | 이번 턴 이후 갱신된 이력서 |
| `reply_done` | — | 턴 종료. TTS 버퍼를 flush하세요 |
| `error` | `msg` | 처리 중 오류 |

---

### Flutter 구현 시 참고

1. **연결 순서**: `connect` → `ready` 수신 → `sync_resume` 전송
2. **녹음**: 버튼 누름 시 녹음 시작, 떼는 순간 `turn` JSON + audio binary 순서로 전송
3. **TTS**: `llm_token`을 받을 때마다 큐에 추가 → 문장 종결부호 감지 시 재생, `reply_done` 수신 시 남은 버퍼 강제 재생
4. **이력서 UI 갱신**: `resume_state`와 `resume_updated` 수신 시 화면 업데이트
5. **오디오 포맷**: `MediaRecorder`(웹) 또는 `flutter_sound` 기준 AAC 또는 webm/opus 권장

---

## 오류 코드 정리

| HTTP Status | 의미 |
|---|---|
| 200 | 성공 |
| 400 | 잘못된 요청 (user_id 형식 오류 등) |
| 500 | 서버 내부 오류 (LLM / STT / RAG) |

---

*최종 수정: 2026-05-12*
