"""
server.py — FastAPI 메인

엔드포인트:
  GET    /health
  POST   /refresh                  100세누리 채용공고 수동 재수집
  WS     /voice/ws                 ★ 메인: 음성 챗 + 이력서 누적 (user_id 기반)
  GET    /resume/{user_id}         현재 이력서 + 빈 항목
  POST   /resume/{user_id}         이력서 필드 직접 수정 (부분 업데이트)
  DELETE /resume/{user_id}         이력서/대화 이력 초기화
  POST   /recommend/{user_id}      저장된 이력서로 채용 추천
  POST   /recommend                ResumeData 직접 받아 추천

TTS는 서버에서 제거됨 — 앱(Flutter)의 디바이스 TTS가 llm_token을 받아 직접 처리.
"""
import os
import json
import asyncio
import tempfile
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import llm
import rag
import stt
import resumes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모든 모듈 초기화 (lifespan 패턴)"""
    if not llm.check_claude():
        raise RuntimeError("Claude API 키 없음. ANTHROPIC_API_KEY 환경변수를 설정하세요.")
    rag.setup()
    stt.setup()
    yield


app = FastAPI(title="Silver Voice Resume API", version="0.2.0", lifespan=lifespan)

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


class ResumeUpdateRequest(BaseModel):
    name: str | None = None
    age: int | None = None
    location: str | None = None
    career: str | None = None
    preferred_work_type: str | None = None
    physical_condition: str | None = None


class RecommendRequest(BaseModel):
    resume: ResumeData
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "model": llm.MODEL_NAME}


@app.post("/refresh")
def refresh_jobs():
    """100세누리 채용공고 수동 재수집 → ChromaDB upsert."""
    try:
        before = rag._collection.count()
        rag.refresh()
        after = rag._collection.count()
        return {"status": "ok", "before": before, "after": after, "delta": after - before}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/voice/ws")
async def voice_ws(ws: WebSocket):
    """
    음성 챗 + 이력서 누적. TTS는 앱 디바이스에서 처리.

    클라 → 서버
        text JSON: {"type":"sync_resume", "user_id":"..."}   ← 현재 상태 조회
        text JSON: {"type":"turn", "user_id":"..."}          ← 새 발화 시작
        binary   : <audio>                                   ← turn 직후 1회 (AAC/webm 등)

    서버 → 클라  (binary 전송 없음 — 텍스트 JSON만)
        {"type":"ready"}                                     ← 연결 직후 1회
        {"type":"resume_state", "resume":{...}, "missing_korean":[...]}
        {"type":"transcript", "text":"..."}                  ← STT 결과
        {"type":"llm_token", "text":"..."}                   ← LLM 토큰 스트림 (N회)
        {"type":"resume_updated", "resume":{...}, "missing_korean":[...]}
        {"type":"reply_done"}                                ← 턴 종료
        {"type":"error", "msg":"..."}
    """
    await ws.accept()
    await ws.send_json({"type": "ready"})
    await ws.send_json({
        "type": "greeting",
        "text": (
            "안녕하세요! 시니어 일자리 상담 도우미입니다. "
            "버튼을 누르고 계신 동안 말씀해 주세요. "
            "이름, 나이, 사시는 곳, 하셨던 일, 원하시는 근무 형태, 건강 상태를 여쭤볼게요. "
            "준비되시면 버튼을 눌러 시작해 주세요."
        ),
    })

    while True:
        try:
            init = await ws.receive_json()
            msg_type = init.get("type")
            user_id = (init.get("user_id") or "").strip()
            if not user_id:
                await ws.send_json({"type": "error", "msg": "user_id required"})
                continue

            # ── 분기 1: sync 요청 ──
            if msg_type == "sync_resume":
                try:
                    snapshot = resumes.public_view(resumes.load(user_id))
                    await ws.send_json({"type": "resume_state", **snapshot})
                except Exception as e:
                    await ws.send_json({"type": "error", "msg": str(e)})
                continue

            if msg_type != "turn":
                await ws.send_json({"type": "error", "msg": "expected type=turn or sync_resume"})
                continue

            # ── 분기 2: 일반 턴 ──

            # 1) 오디오 받기
            audio_bytes = await ws.receive_bytes()

            # 2) 유저 상태 로드 + 턴 시작 스냅샷 push
            resume = resumes.load(user_id)
            history = list(resume.get("history", []))
            await ws.send_json({"type": "resume_state", **resumes.public_view(resume)})

            # 3) STT (ffmpeg → WAV → Whisper)
            in_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                    f.write(audio_bytes)
                    in_path = f.name
                user_text = await asyncio.to_thread(stt.transcribe, in_path)
            finally:
                if in_path and os.path.exists(in_path):
                    os.unlink(in_path)

            await ws.send_json({"type": "transcript", "text": user_text})

            # 4) 동적 system prompt
            missing = resumes.missing_fields(resume)
            sys_prompt = llm.build_voice_system_prompt(resume, missing)

            # 5) LLM stream → 토큰 전송 (TTS는 앱에서 처리)
            full_reply = ""
            async for token in llm.chat_stream_with_prompt(sys_prompt, history, user_text):
                await ws.send_json({"type": "llm_token", "text": token})
                full_reply += token

            # 6) history 업데이트
            full_reply = llm.strip_markdown(full_reply)
            resumes.append_turn(resume, user_text, full_reply)

            # 7) 이력서 추출 (백그라운드 — 사용자 체감 지연 없음)
            updated = await asyncio.to_thread(llm.extract_resume, resume["history"], resume)
            updated["history"] = resume["history"]
            resumes.save(user_id, updated)

            # 8) 변경 후 상태 + 턴 종료 push
            await ws.send_json({"type": "resume_updated", **resumes.public_view(updated)})
            await ws.send_json({"type": "reply_done"})

        except WebSocketDisconnect:
            print("WS 연결 종료")
            return
        except Exception as e:
            print(f"WS 오류: {e!r}")
            try:
                await ws.send_json({"type": "error", "msg": str(e)})
            except Exception:
                return


@app.get("/resume/{user_id}")
def get_resume(user_id: str):
    try:
        return resumes.public_view(resumes.load(user_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/resume/{user_id}")
def reset_resume(user_id: str):
    try:
        resumes.reset(user_id)
        return {"status": "ok", "user_id": user_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/resume/{user_id}")
def update_resume(user_id: str, req: ResumeUpdateRequest):
    """프론트에서 직접 이력서 필드를 수정할 때 사용."""
    try:
        resume = resumes.load(user_id)
        update = req.model_dump(exclude_none=True)
        resume.update(update)
        resumes.save(user_id, resume)
        return resumes.public_view(resume)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/recommend/{user_id}")
def recommend_by_user(user_id: str, top_k: int = 5):
    """저장된 이력서로 추천."""
    try:
        resume = resumes.load(user_id)
        miss = resumes.missing_fields(resume)
        if miss:
            return {
                "recommendations": [],
                "message": "이력서가 아직 미완성입니다. 음성 채팅으로 채워주세요.",
                "missing": [resumes.FIELD_LABELS[k] for k in miss],
            }
        resume_dict = {k: resume[k] for k in resumes.FIELD_LABELS}
        jobs = rag.search(resume_dict, top_k=top_k)
        if not jobs:
            return {"recommendations": [], "message": "조건에 맞는 채용공고가 없습니다."}
        prompt = rag.build_recommend_prompt(resume_dict, jobs)
        llm_response = llm.chat(prompt)
        try:
            start = llm_response.find("{")
            end = llm_response.rfind("}") + 1
            return json.loads(llm_response[start:end])
        except json.JSONDecodeError:
            return {"recommendations": jobs, "raw_llm": llm_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """이력서 → RAG 검색 → LLM 추천 이유 생성 → JSON"""
    try:
        resume_dict = req.resume.model_dump()
        jobs = rag.search(resume_dict, top_k=req.top_k)
        if not jobs:
            return {"recommendations": [], "message": "조건에 맞는 채용공고가 없습니다."}
        prompt = rag.build_recommend_prompt(resume_dict, jobs)
        llm_response = llm.chat(prompt)
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
