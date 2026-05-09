"""
resumes.py — 유저별 이력서 + 대화 이력 저장소

저장 위치: ./resumes/{user_id}.json (RunPod 볼륨에 영구)
파일당 한 사용자, JSON 한 덩어리.
"""
import json
import os
import threading
from pathlib import Path

RESUMES_DIR = "./resumes"
HISTORY_CAP = 20  # 최근 N턴만 보관 (LLM 컨텍스트 폭주 방지)

DEFAULT_RESUME = {
    "name": "",
    "age": 0,
    "location": "",
    "career": "",
    "preferred_work_type": "",
    "physical_condition": "",
    "history": [],  # [{"role":"user"/"assistant","content":"..."}]
}

# 항목 한글 라벨 (LLM 프롬프트/UI 공유)
FIELD_LABELS = {
    "name": "이름",
    "age": "나이",
    "location": "거주지",
    "career": "경력",
    "preferred_work_type": "희망 근무형태",
    "physical_condition": "건강 상태/체력",
}

_lock = threading.Lock()
Path(RESUMES_DIR).mkdir(exist_ok=True)


def _path(user_id: str) -> str:
    safe = "".join(c for c in user_id if c.isalnum() or c in "-_")
    if not safe:
        raise ValueError("invalid user_id")
    return os.path.join(RESUMES_DIR, f"{safe}.json")


def load(user_id: str) -> dict:
    """없으면 DEFAULT_RESUME 사본 반환 (디스크에는 안 만듦, save() 시 생성)"""
    path = _path(user_id)
    if not os.path.exists(path):
        return {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_RESUME.items()}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 누락된 키 채워서 후방 호환
    for k, v in DEFAULT_RESUME.items():
        if k not in data:
            data[k] = list(v) if isinstance(v, list) else v
    return data


def save(user_id: str, resume: dict):
    """원자적 쓰기 (tmp → rename) + 잠금"""
    path = _path(user_id)
    tmp = path + ".tmp"
    with _lock:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(resume, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


def reset(user_id: str):
    """유저 이력서/이력 삭제"""
    path = _path(user_id)
    if os.path.exists(path):
        os.remove(path)


def missing_fields(resume: dict) -> list[str]:
    """비어있는 항목 키 리스트 (FIELD_LABELS 기준)"""
    out = []
    for key in FIELD_LABELS:
        v = resume.get(key)
        if key == "age":
            if not isinstance(v, int) or v <= 0:
                out.append(key)
        else:
            if not (isinstance(v, str) and v.strip()):
                out.append(key)
    return out


def append_turn(resume: dict, user_text: str, assistant_text: str):
    """history 끝에 turn 추가 + cap 적용 (in-place)"""
    history = resume.get("history", [])
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    resume["history"] = history[-HISTORY_CAP * 2:]  # 한 turn = 2 메시지


def public_view(resume: dict) -> dict:
    """클라에 노출할 부분만 (history 제외) + 누락 항목 같이"""
    return {
        "resume": {k: resume.get(k, DEFAULT_RESUME[k]) for k in FIELD_LABELS},
        "missing": missing_fields(resume),
        "missing_korean": [FIELD_LABELS[k] for k in missing_fields(resume)],
    }
