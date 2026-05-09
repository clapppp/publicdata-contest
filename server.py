"""
server.py — FastAPI 메인
/voice      : (POST)  오디오 → STT → LLM → TTS → JSON (legacy, 음성메시지)
/voice/ws   : (WS)    실시간 스트리밍 (Stage 2: push-to-talk + LLM/TTS 토큰 스트림)
/recommend  : (POST)  이력서 → RAG → LLM → JSON
/refresh    : (POST)  100세누리 채용공고 재수집
/health     : (GET)   상태 확인
"""
import os
import json
import asyncio
import base64
import tempfile
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
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


@app.post("/refresh")
def refresh_jobs():
    """
    워크넷 API에서 채용공고 재수집 → ChromaDB upsert.
    수동 호출 또는 일일 06시 스케줄러에서 사용.
    """
    try:
        before = rag._collection.count()
        rag.refresh()
        after = rag._collection.count()
        return {
            "status": "ok",
            "before": before,
            "after": after,
            "delta": after - before,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


SENTENCE_END = ".?!。？！"


def _is_sentence_end(buf: str) -> bool:
    """문장 끝났는지 단순 휴리스틱: rstrip 후 마지막 문자가 종결 부호이고 길이 충분"""
    s = buf.rstrip()
    return bool(s) and s[-1] in SENTENCE_END and len(s) > 5


@app.websocket("/voice/ws")
async def voice_ws(ws: WebSocket):
    """
    Stage 2 스트리밍 음성 챗 (push-to-talk).

    프로토콜:
        클라 → 서버
            text JSON: {"type":"turn", "history":[{role,content},...]}
            binary  : <audio file (mp3/wav/webm/m4a/...) bytes>
        서버 → 클라
            text JSON: {"type":"ready"}                    (연결 직후 1회)
            text JSON: {"type":"transcript","text":"..."}  (STT 결과)
            text JSON: {"type":"llm_token","text":"..."}   (LLM 토큰 N회)
            binary  : <mp3 chunk> N회 (TTS 음성)
            text JSON: {"type":"reply_done"}               (한 턴 응답 종료)
            text JSON: {"type":"error","msg":"..."}        (오류 시)

    한 WS 연결로 여러 턴 가능 — history는 매 턴마다 클라가 누적해서 보냄 (서버 stateless).
    """
    await ws.accept()
    await ws.send_json({"type": "ready"})

    while True:
        try:
            # 1) turn 시작 (history 받음)
            init = await ws.receive_json()
            if init.get("type") != "turn":
                await ws.send_json({"type": "error", "msg": "expected type=turn"})
                continue
            history = init.get("history", [])

            # 2) 오디오 받기
            audio_bytes = await ws.receive_bytes()

            # 3) STT (file로 떨어뜨리고 기존 transcribe 호출, blocking이라 to_thread)
            in_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
                    f.write(audio_bytes)
                    in_path = f.name
                user_text = await asyncio.to_thread(stt.transcribe, in_path)
            finally:
                if in_path and os.path.exists(in_path):
                    os.unlink(in_path)

            await ws.send_json({"type": "transcript", "text": user_text})

            # 4) LLM 토큰 스트림 → 문장 단위로 모아 TTS 청크 흘리기
            sentence_buf = ""
            async for token in llm.chat_stream("voice", user_text, history):
                await ws.send_json({"type": "llm_token", "text": token})
                sentence_buf += token

                if _is_sentence_end(sentence_buf):
                    async for chunk in tts.synthesize_stream(sentence_buf):
                        await ws.send_bytes(chunk)
                    sentence_buf = ""

            # 5) 남은 텍스트 마저 TTS
            if sentence_buf.strip():
                async for chunk in tts.synthesize_stream(sentence_buf):
                    await ws.send_bytes(chunk)

            await ws.send_json({"type": "reply_done"})

        except WebSocketDisconnect:
            print("WS 연결 종료")
            return
        except Exception as e:
            print(f"WS 오류: {e!r}")
            try:
                await ws.send_json({"type": "error", "msg": str(e)})
            except Exception:
                return  # 클라 이미 끊김
            # 오류는 보고만 하고 다음 턴 대기 (연결 유지)


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
