"""
tts.py — Coqui XTTS v2 Korean (로컬 GPU)

MeloTTS에서 전환. XTTS v2는 multilingual 고품질 모델로 한국어 자연스러움이 크게 향상됨.
모델 크기 ~1.8GB (HF_HOME 캐시), VRAM ~2GB.

- setup()              : 모델 로드 (서버 시작 시 1회 호출)
- synthesize()         : 동기, 텍스트 → wav 파일 저장 (legacy POST /voice 용)
- synthesize_stream()  : 비동기 generator, wav bytes 한 청크 yield (/voice/ws 용)

출력 포맷: WAV 24kHz (XTTS v2 네이티브). 한 문장 = 한 wav 청크.
클라(test_client.html)가 청크별로 순차 재생하는 구조는 MeloTTS와 동일.
"""
import asyncio
import io

import numpy as np
import soundfile as sf

# Lazy-loaded
_model = None
_SAMPLE_RATE = 24000
# XTTS v2 내장 스피커 — model.speakers 로 전체 목록 확인 가능
# 한국어 발음이 자연스러운 스피커 (기본값: 여성)
_SPEAKER = "Claribel Dervla"


def setup():
    """Coqui XTTS v2 모델 로드. 서버 lifespan에서 1회 호출."""
    global _model
    if _model is not None:
        return

    print("📦 Coqui XTTS v2 모델 로드 중 (GPU)")
    print("   ↳ 첫 실행 시 ~1.8GB HF 다운로드 (이후엔 HF_HOME 캐시)")

    from TTS.api import TTS  # pip install TTS

    _model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    print(f"   ↳ XTTS v2 준비 완료 (sample_rate={_SAMPLE_RATE}Hz, speaker={_SPEAKER})")


def _synth_bytes(text: str, speed: float = 1.0) -> bytes:
    """동기 합성 → WAV bytes (메모리 내, 임시파일 불필요)."""
    if _model is None:
        raise RuntimeError("tts.setup()을 먼저 호출하세요.")

    # tts() → list[float] (24kHz PCM)
    wav = _model.tts(text=text, speaker=_SPEAKER, language="ko")
    wav_np = np.array(wav, dtype=np.float32)

    # speed 조정 (1.0 이외: 피치 보존 time-stretch)
    if abs(speed - 1.0) > 0.01:
        try:
            import librosa
            wav_np = librosa.effects.time_stretch(wav_np, rate=speed)
        except ImportError:
            pass  # librosa 없으면 속도 조정 생략

    buf = io.BytesIO()
    sf.write(buf, wav_np, _SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def synthesize(text: str, out_path: str, speed: float = 1.0) -> str:
    """텍스트 → WAV 파일 저장. legacy POST /voice 용."""
    data = _synth_bytes(text, speed)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


async def synthesize_stream(text: str, speed: float = 1.0):
    """
    텍스트 → WAV bytes async generator.

    XTTS v2는 문장 단위 일괄 합성이라 청크가 1개 — 한 번 yield 후 종료.
    server.py가 문장별로 호출하므로 결과적으로 문장당 1 wav 청크가 클라로 전송됨.
    """
    data = await asyncio.to_thread(_synth_bytes, text, speed)
    yield data


if __name__ == "__main__":
    setup()
    synthesize("안녕하세요. XTTS v2 한국어 음성 테스트입니다.", "test.wav")
    print("✅ test.wav 생성")

    async def _stream_demo():
        chunks = []
        async for c in synthesize_stream("스트리밍 테스트도 동일한 인터페이스로 동작합니다."):
            chunks.append(c)
        with open("test_stream.wav", "wb") as f:
            f.write(b"".join(chunks))
        print(f"✅ test_stream.wav 생성 ({len(chunks)} chunks, {sum(len(c) for c in chunks)} bytes)")

    asyncio.run(_stream_demo())
