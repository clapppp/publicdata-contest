"""
server.py — FastAPI 메인
/voice     : 오디오 입력 → STT → LLM → TTS → JSON (음성 base64)
/recommend : 이력서 → RAG → LLM → JSON
/health    : 상태 확인
"""
import os
import json
import base64
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import llm
import rag
import stt
import tts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모든 모듈 초기화 (lifespan 패턴)"""
    if not llm.check_ollama():
        raise RuntimeError("Ollama 미준비. setup.sh + load_model.sh 실행 필요")
    rag.setup()
    stt.setup()
    yield


app = FastAPI(title="Silver Voice Resume API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResumeData(BaseModel):
    name: str = ""
    age: int = 0
    location: str = ""
    career: str = ""
    preferred_work_type: str = ""
    physical_condition: str = ""


class RecommendRequest(BaseModel):
    resume: ResumeData
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "model": llm.MODEL_NAME}


@app.post("/voice")
async def voice_chat(
    audio: UploadFile = File(...),
    history: str = Form("[]"),
):
    """
    오디오 입력 → STT → LLM → TTS → 음성 응답

    Form fields:
        audio: 사용자 음성 파일 (wav/mp3/m4a/webm 모두 가능)
        history: 이전 대화 이력 (JSON 문자열, 기본 "[]")

    Returns:
        user_text: STT 결과
        reply_text: LLM 응답 텍스트
        reply_audio_b64: TTS 음성 (mp3, base64)
        history: 다음 턴용 누적 이력
    """
    in_path = out_path = None
    try:
        history_list = json.loads(history)

        # 1) STT
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(await audio.read())
            in_path = f.name
        user_text = stt.transcribe(in_path)

        # 2) LLM
        reply_text = llm.chat(role="voice", user_message=user_text, history=history_list)

        # 3) TTS
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            out_path = f.name
        tts.synthesize(reply_text, out_path)
        with open(out_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        return {
            "user_text": user_text,
            "reply_text": reply_text,
            "reply_audio_b64": audio_b64,
            "history": history_list + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply_text},
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                os.unlink(p)


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    이력서 → RAG 검색 → LLM 추천 이유 생성 → JSON
    """
    try:
        resume_dict = req.resume.model_dump()

        jobs = rag.search(resume_dict, top_k=req.top_k)
        if not jobs:
            return {"recommendations": [], "message": "조건에 맞는 채용공고가 없습니다."}

        prompt = rag.build_recommend_prompt(resume_dict, jobs)
        llm_response = llm.chat(role="recommend", user_message=prompt)

        # LLM이 JSON 외 텍스트를 붙이는 경우 방어
        try:
            start = llm_response.find("{")
            end = llm_response.rfind("}") + 1
            return json.loads(llm_response[start:end])
        except json.JSONDecodeError:
            return {"recommendations": jobs, "raw_llm": llm_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
