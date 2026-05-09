"""
server.py — FastAPI 메인

엔드포인트:
  GET    /health
  POST   /refresh                  100세누리 채용공고 수동 재수집
  POST   /voice                    legacy 음성메시지 (POST, base64 mp3 응답)
  WS     /voice/ws                 ★ 메인: 음성 챗 + 이력서 누적 (user_id 기반)
  GET    /resume/{user_id}         현재 이력서 + 빈 항목
  DELETE /resume/{user_id}         이력서/대화 이력 초기화
  POST   /recommend                ResumeData 직접 받아 추천
  POST   /recommend/{user_id}      저장된 이력서로 추천
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
import resumes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모든 모듈 초기화 (lifespan 패턴)"""
    if not llm.check_ollama():
        raise RuntimeError("Ollama 미준비. setup.sh + load_model.sh 실행 필요")
    rag.setup()
    stt.setup()
    tts.setup()  # MeloTTS 모델 로드 (~500MB GPU, 5-15초)
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

        # 3) TTS (MeloTTS 출력은 wav)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
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
    음성 챗 + 이력서 누적 (Stage 2 스트리밍, 서버 상태 보유).

    클라 → 서버
        text JSON: {"type":"sync_resume", "user_id":"..."}   ← 현재 상태 조회만
        text JSON: {"type":"turn", "user_id":"..."}          ← 새 발화 시작
        binary   : <audio file>                              ← turn 직후 1회

    서버 → 클라
        text JSON: {"type":"ready"}                          ← 연결 직후 1회
        text JSON: {"type":"resume_state", ...}              ← sync 응답 + 턴 시작 스냅샷
        text JSON: {"type":"transcript", "text":"..."}
        text JSON: {"type":"llm_token", "text":"..."}
        binary   : <mp3 chunk>...                             ← 문장 단위 TTS
        text JSON: {"type":"resume_updated", ...}            ← 추출 후 변경된 상태
        text JSON: {"type":"reply_done"}                     ← 한 턴 종료
        text JSON: {"type":"error", "msg":"..."}
    """
    await ws.accept()
    await ws.send_json({"type": "ready"})

    while True:
        try:
            init = await ws.receive_json()
            msg_type = init.get("type")
            user_id = (init.get("user_id") or "").strip()
            if not user_id:
                await ws.send_json({"type": "error", "msg": "user_id required"})
                continue

            # ── 분기 1: 단순 sync 요청 (turn 아님) ──
            if msg_type == "sync_resume":
                try:
                    snapshot = resumes.public_view(resumes.load(user_id))
                    await ws.send_json({"type": "resume_state", **snapshot})
                except Exception as e:
                    await ws.send_json({"type": "error", "msg": str(e)})
                continue

            # ── 분기 2: 일반 턴 ──
            if msg_type != "turn":
                await ws.send_json({"type": "error", "msg": "expected type=turn or sync_resume"})
                continue

            # 1) 오디오 받기
            audio_bytes = await ws.receive_bytes()

            # 2) 유저 상태 로드 + 즉시 스냅샷 push (턴 시작 시점)
            resume = resumes.load(user_id)
            history = list(resume.get("history", []))
            await ws.send_json({"type": "resume_state", **resumes.public_view(resume)})

            # 3) STT
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

            # 4) 동적 system prompt (이력서 상태 + 빈 항목 반영)
            missing = resumes.missing_fields(resume)
            sys_prompt = llm.build_voice_system_prompt(resume, missing)

            # 5) LLM stream → 문장 단위 TTS stream
            sentence_buf = ""
            full_reply = ""
            async for token in llm.chat_stream_with_prompt(sys_prompt, history, user_text):
                await ws.send_json({"type": "llm_token", "text": token})
                sentence_buf += token
                full_reply += token

                if _is_sentence_end(sentence_buf):
                    async for chunk in tts.synthesize_stream(sentence_buf):
                        await ws.send_bytes(chunk)
                    sentence_buf = ""

            if sentence_buf.strip():
                async for chunk in tts.synthesize_stream(sentence_buf):
                    await ws.send_bytes(chunk)

            # 6) history 업데이트
            resumes.append_turn(resume, user_text, full_reply)

            # 7) 이력서 추출 (음성 응답 이후 → 사용자 체감 지연 없음)
            updated = await asyncio.to_thread(llm.extract_resume, resume["history"], resume)
            updated["history"] = resume["history"]
            resumes.save(user_id, updated)

            # 8) 변경 후 상태 push
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
            # 오류 알리고 다음 턴 대기


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


@app.post("/recommend/{user_id}")
def recommend_by_user(user_id: str, top_k: int = 5):
    """저장된 이력서로 추천. /recommend는 ResumeData 직접 받는 버전(legacy)."""
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
        llm_response = llm.chat(role="recommend", user_message=prompt)
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
