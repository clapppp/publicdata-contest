"""
stt.py — Whisper STT (faster-whisper, large-v3, GPU)
"""
from faster_whisper import WhisperModel

MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"  # 4090에 적합. VRAM 부족 시 "int8_float16"

_model = None


def setup():
    """Whisper 모델 로드 (서버 시작 시 1회)"""
    global _model
    print(f"📦 Whisper {MODEL_SIZE} 로드 중 ({DEVICE}/{COMPUTE_TYPE})")
    _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("✅ Whisper 준비 완료")


def transcribe(audio_path: str, language: str = "ko") -> str:
    """오디오 파일 → 텍스트"""
    if _model is None:
        raise RuntimeError("setup()을 먼저 호출하세요.")
    segments, _ = _model.transcribe(audio_path, language=language, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments)


if __name__ == "__main__":
    import sys
    setup()
    if len(sys.argv) > 1:
        print(transcribe(sys.argv[1]))
