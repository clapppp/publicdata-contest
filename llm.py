"""
llm.py — Claude API 클라이언트 (claude-haiku-4-5)

함수 종류:
- check_claude()                   : API 키 / 연결 확인
- chat()                           : 동기, 전체 응답 한 번 (legacy POST /voice, /recommend)
- chat_stream_with_prompt()        : 비동기, system prompt 직접 주입 stream (/voice/ws)
- chat_stream()                    : 비동기, role 기반 stream (legacy)
- build_voice_system_prompt()      : 이력서 상태 → 동적 voice system prompt
- extract_resume()                 : 동기, 대화 history → 구조화된 ResumeData JSON
"""
import json
import re
import anthropic

MODEL_NAME = "claude-haiku-4-5"

_client = anthropic.Anthropic()           # 동기 호출용 (chat, extract_resume, check)
_async_client = anthropic.AsyncAnthropic()  # 비동기 스트리밍용 (voice/ws)

SYSTEM_PROMPTS = {
    "voice": """너는 시니어 구직자를 돕는 친절한 상담사야.
한 번에 한 가지 질문만 해야 해. 짧고 쉬운 말로 대화해.
이름, 나이, 주소, 연락처, 경력, 학력 순서로 물어봐.
사용자가 답하면 다음 항목을 물어봐.""",

    "recommend": """너는 시니어 구직 매칭 전문가야.
아래 이력서 정보와 채용공고를 분석해서 반드시 JSON 형식으로만 반환해.
다른 말은 절대 하지 마. JSON만 반환해.
형식:
{
  "recommendations": [
    {
      "job_id": "공고ID",
      "job_title": "직종명",
      "company": "회사명",
      "score": 85,
      "reason": "추천 이유 한 문장"
    }
  ]
}""",
}


def check_claude() -> bool:
    """Claude API 키 및 연결 확인"""
    try:
        _client.messages.create(
            model=MODEL_NAME,
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
        )
        print(f"✅ Claude API 연결 OK | 모델: {MODEL_NAME}")
        return True
    except anthropic.AuthenticationError:
        print("❌ Claude API 키 오류. ANTHROPIC_API_KEY 환경변수를 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ Claude API 연결 실패: {e}")
        return False


def chat(role: str, user_message: str, history: list | None = None) -> str:
    """
    동기 단일 응답.
    Args:
        role: "voice" 또는 "recommend"
        user_message: 사용자 입력
        history: [{"role": "user"/"assistant", "content": "..."}]
    Returns:
        LLM 응답 텍스트
    """
    if history is None:
        history = []

    messages = list(history)
    messages.append({"role": "user", "content": user_message})

    response = _client.messages.create(
        model=MODEL_NAME,
        max_tokens=2048,
        system=SYSTEM_PROMPTS[role],
        messages=messages,
        temperature=0.7 if role == "voice" else 0.1,
    )
    return response.content[0].text


# ── 동적 voice prompt (이력서 상태 기반) ────────────────────────────────────

VOICE_SYSTEM_TEMPLATE = """너는 시니어 구직자를 돕는 친절한 음성 상담사야.
짧고 쉬운 말로 대화해. 어려운 단어 쓰지 마. 한 번에 한 가지 질문만 해.

[지금까지 알아낸 이력서]
- 이름: {name}
- 나이: {age}
- 거주지: {location}
- 경력: {career}
- 희망 근무형태: {preferred_work_type}
- 건강 상태: {physical_condition}

[아직 비어있는 항목]
{missing_korean}

[규칙]
1. 사용자가 방금 한 말에 짧게 반응해 ("네, 알겠습니다" 등)
2. 그 다음 비어있는 항목 중 하나만 골라 자연스럽게 물어봐
3. 이미 채워진 항목은 다시 묻지 마
4. 모든 항목이 채워졌으면 "이력서 작성이 다 끝났어요. 일자리 추천을 받아보시겠어요?" 라고 마무리해"""


VOICE_SYSTEM_COMPLETE = """너는 시니어 구직자를 돕는 친절한 음성 상담사야.
사용자 이력서가 모두 채워진 상태야. 짧게 인사하고 "일자리 추천을 받아보시겠어요?" 라고 마무리해.
새 질문은 더 안 해."""


def build_voice_system_prompt(resume: dict, missing_keys: list[str]) -> str:
    """이력서 상태와 빈 항목 → voice 모드 system prompt"""
    field_labels = {
        "name": "이름", "age": "나이", "location": "거주지",
        "career": "경력", "preferred_work_type": "희망 근무형태",
        "physical_condition": "건강 상태/체력",
    }

    if not missing_keys:
        return VOICE_SYSTEM_COMPLETE

    def show(key, default="(미입력)"):
        v = resume.get(key)
        if key == "age":
            return v if isinstance(v, int) and v > 0 else default
        return v if isinstance(v, str) and v.strip() else default

    return VOICE_SYSTEM_TEMPLATE.format(
        name=show("name"),
        age=show("age"),
        location=show("location"),
        career=show("career"),
        preferred_work_type=show("preferred_work_type"),
        physical_condition=show("physical_condition"),
        missing_korean=", ".join(field_labels[k] for k in missing_keys),
    )


# ── 이력서 추출 (history → ResumeData JSON) ────────────────────────────────

EXTRACT_SYSTEM = """너는 대화에서 시니어 구직자의 이력서 정보를 추출하는 분석가야.
아래 대화를 분석해서 사용자가 말한 이력서 정보를 JSON으로만 반환해.

[기존 이력서]
{existing_json}

[규칙]
1. JSON 외에 다른 텍스트는 절대 출력하지 마.
2. 대화에서 새로 알게 된 정보가 있으면 그 필드를 채우고, 변경 정보 없는 필드는 기존 값 그대로 유지.
3. 나이는 정수. 나머지는 문자열.
4. 알 수 없으면 빈 문자열 "" 또는 0.

[JSON 스키마 — 정확히 이 키들만 사용]
{{
  "name": "",
  "age": 0,
  "location": "",
  "career": "",
  "preferred_work_type": "",
  "physical_condition": ""
}}"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_resume(history: list, existing: dict) -> dict:
    """
    대화 history → 업데이트된 이력서 dict 반환.
    동기 호출 (background에서 호출 권장).
    실패 시 existing 그대로 반환 (안전 폴백).
    """
    fields = ["name", "age", "location", "career", "preferred_work_type", "physical_condition"]
    existing_clean = {k: existing.get(k, 0 if k == "age" else "") for k in fields}
    existing_json = json.dumps(existing_clean, ensure_ascii=False)

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[-20:]
    )

    response = _client.messages.create(
        model=MODEL_NAME,
        max_tokens=512,
        system=EXTRACT_SYSTEM.format(existing_json=existing_json),
        messages=[{"role": "user", "content": history_text}],
        temperature=0.0,
    )
    raw = response.content[0].text
    cleaned = _THINK_RE.sub("", raw).strip()

    # JSON 추출
    try:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start < 0 or end <= start:
            return existing_clean | {k: existing.get(k) for k in existing if k not in fields}
        parsed = json.loads(cleaned[start:end])
    except (json.JSONDecodeError, ValueError):
        return existing_clean | {k: existing.get(k) for k in existing if k not in fields}

    # 머지: 새 값이 비어있으면 기존 유지
    merged = dict(existing)
    for key in fields:
        new = parsed.get(key)
        if key == "age":
            if isinstance(new, int) and new > 0:
                merged[key] = new
        else:
            if isinstance(new, str) and new.strip():
                merged[key] = new.strip()
    return merged


# ── 스트리밍: system prompt 직접 주입 (/voice/ws 전용) ──────────────────────

async def chat_stream_with_prompt(
    system_prompt: str,
    history: list,
    user_message: str,
    temperature: float = 0.7,
):
    """
    Claude API 스트리밍. system prompt를 외부에서 주입.
    voice/ws에서 매 턴마다 이력서 상태 반영한 동적 prompt에 사용.

    Yields:
        str — 텍스트 토큰 조각
    """
    messages = list(history)
    messages.append({"role": "user", "content": user_message})

    async with _async_client.messages.stream(
        model=MODEL_NAME,
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
        temperature=temperature,
    ) as stream:
        async for text in stream.text_stream:
            yield text


# ── legacy stream (role 기반) ────────────────────────────────────────────────

async def chat_stream(role: str, user_message: str, history: list | None = None):
    """
    role 기반 스트리밍 (legacy — /voice/ws 이전 호환용).

    Yields:
        str — 텍스트 토큰 조각
    """
    if history is None:
        history = []

    messages = list(history)
    messages.append({"role": "user", "content": user_message})

    async with _async_client.messages.stream(
        model=MODEL_NAME,
        max_tokens=1024,
        system=SYSTEM_PROMPTS[role],
        messages=messages,
        temperature=0.7 if role == "voice" else 0.1,
    ) as stream:
        async for text in stream.text_stream:
            yield text


if __name__ == "__main__":
    if check_claude():
        print("\n--- voice 모드 테스트 (sync) ---")
        print(chat("voice", "안녕하세요, 이력서 작성 도와주세요"))

        print("\n--- voice 모드 테스트 (stream) ---")
        import asyncio

        async def _demo():
            async for token in chat_stream("voice", "안녕하세요, 이력서 작성 도와주세요"):
                print(token, end="", flush=True)
            print()

        asyncio.run(_demo())
