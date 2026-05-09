"""
llm.py — Ollama 클라이언트 (qwen3 8b)
system prompt로 voice / recommend 역할 분기
"""
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:8b"  # load_model.sh로 등록되는 이름

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


def check_ollama() -> bool:
    """Ollama 서버 + 모델 등록 확인"""
    try:
        res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in res.json().get("models", [])]
        if not any(MODEL_NAME in m for m in models):
            print(f"❌ {MODEL_NAME} 미등록. load_model.sh를 먼저 실행하세요.")
            return False
        print(f"✅ Ollama 연결 OK | 모델: {MODEL_NAME}")
        return True
    except Exception as e:
        print(f"❌ Ollama 연결 실패: {e}")
        return False


def chat(role: str, user_message: str, history: list | None = None) -> str:
    """
    Args:
        role: "voice" 또는 "recommend"
        user_message: 사용자 입력
        history: [{"role": "user"/"assistant", "content": "..."}]
    Returns:
        LLM 응답 텍스트
    """
    if history is None:
        history = []

    messages = [{"role": "system", "content": SYSTEM_PROMPTS[role]}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    res = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7 if role == "voice" else 0.1,
                "num_ctx": 8192,
            },
        },
        timeout=120,
    )
    return res.json()["message"]["content"]


if __name__ == "__main__":
    if check_ollama():
        print("\n--- voice 모드 테스트 ---")
        print(chat("voice", "안녕하세요, 이력서 작성 도와주세요"))
