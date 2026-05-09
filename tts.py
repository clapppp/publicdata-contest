"""
tts.py — MeloTTS Korean (로컬 GPU)

edge-tts에서 전환 (RunPod IP를 Microsoft가 403 거절). 로컬 모델이라 인터넷 불필요.

- setup()              : 모델 로드 (서버 시작 시 1회 호출)
- synthesize()         : 동기, 텍스트 → wav 파일 저장 (legacy POST /voice 용)
- synthesize_stream()  : 비동기 generator, wav bytes 한 청크 yield (/voice/ws 용)

출력 포맷: WAV (MeloTTS 네이티브). 한 문장 = 한 wav 파일 = 한 청크.
edge-tts(mp3 부분 청크)와 차이: 클라가 청크별로 순차 재생해야 함 (test_client.html 처리).
"""
import asyncio
import io
import os
import tempfile

import soundfile as sf

# Lazy-loaded
_model = None
_speaker_id = None
_sample_rate = None


def setup():
    """MeloTTS Korean 모델 로드. 서버 lifespan에서 1회 호출."""
    global _model, _speaker_id, _sample_rate
    if _model is not None:
        return

    print("📦 MeloTTS Korean 모델 로드 중 (GPU)")
    print("   ↳ 첫 실행 시 ~500MB HF 다운로드 (이후엔 HF_HOME 캐시)")

    from melo.api import TTS  # 임포트 자체가 무거워서 setup 시점에

    _model = TTS(language="KR", device="cuda")
    _speaker_id = _model.hps.data.spk2id["KR"]
    _sample_rate = _model.hps.data.sampling_rate
    print(f"   ↳ MeloTTS 준비 완료 (sample_rate={_sample_rate}Hz)")


def _synth_bytes(text: str, speed: float = 1.0) -> bytes:
    """동기 합성 → WAV bytes."""
    if _model is None:
        raise RuntimeError("tts.setup()을 먼저 호출하세요.")

    # MeloTTS는 file path 인자를 요구. tempfile 한 사이클.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        _model.tts_to_file(text, _speaker_id, path, speed=speed)
        with open(path, "rb") as fp:
            return fp.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def synthesize(text: str, out_path: str, speed: float = 1.0) -> str:
    """텍스트 → WAV 파일 저장. legacy POST /voice 용."""
    data = _synth_bytes(text, speed)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


async def synthesize_stream(text: str, speed: float = 1.0):
    """
    텍스트 → WAV bytes async generator.

    MeloTTS는 문장 단위 일괄 합성이라 청크가 1개 — 한 번 yield 후 종료.
    server.py가 문장별로 호출하므로 결과적으로 문장당 1 wav 청크가 클라로 전송됨.
    """
    data = await asyncio.to_thread(_synth_bytes, text, speed)
    yield data


if __name__ == "__main__":
    setup()
    synthesize("안녕하세요. MeloTTS 로컬 한국어 음성 테스트입니다.", "test.wav")
    print("✅ test.wav 생성")

    async def _stream_demo():
        chunks = []
        async for c in synthesize_stream("스트리밍 테스트도 동일한 인터페이스로 동작합니다."):
            chunks.append(c)
        with open("test_stream.wav", "wb") as f:
            f.write(b"".join(chunks))
        print(f"✅ test_stream.wav 생성 ({len(chunks)} chunks, {sum(len(c) for c in chunks)} bytes)")

    asyncio.run(_stream_demo())
