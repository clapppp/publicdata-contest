"""
stt.py — Whisper STT (faster-whisper, large-v3, GPU)

브라우저 MediaRecorder는 Chrome=webm/opus, Firefox=ogg/opus, Safari=mp4 등
포맷이 제각각이라 ffmpeg으로 먼저 16kHz mono WAV로 변환 후 Whisper에 넘김.
"""
import os
import subprocess
import tempfile

from faster_whisper import WhisperModel

MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"  # RTX 4000 Ada 적합. VRAM 부족 시 "int8_float16"

_model = None


def setup():
    """Whisper 모델 로드 (서버 시작 시 1회)"""
    global _model
    print(f"📦 Whisper {MODEL_SIZE} 로드 중 ({DEVICE}/{COMPUTE_TYPE})")
    _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("✅ Whisper 준비 완료")


def _to_wav(src: str) -> str:
    """
    ffmpeg으로 임의 포맷 오디오 → 16kHz mono WAV 변환.
    반환: 변환된 WAV 경로 (호출자가 삭제해야 함)
    """
    dst = tempfile.mktemp(suffix=".wav")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", src,
            "-ar", "16000",   # Whisper 권장 샘플레이트
            "-ac", "1",       # mono
            "-f", "wav",
            dst,
        ],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg 변환 실패:\n{err}")
    return dst


def transcribe(audio_path: str, language: str = "ko") -> str:
    """임의 포맷 오디오 파일 → 텍스트 (ffmpeg 전처리 포함)"""
    if _model is None:
        raise RuntimeError("setup()을 먼저 호출하세요.")

    wav_path = None
    try:
        wav_path = _to_wav(audio_path)
        segments, _ = _model.transcribe(wav_path, language=language, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments)
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


if __name__ == "__main__":
    import sys
    setup()
    if len(sys.argv) > 1:
        print(transcribe(sys.argv[1]))
