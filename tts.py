"""
tts.py — Microsoft Edge TTS (한국어 Neural Voice)

장점: 한국어 자연스러움 최상, 의존성 가벼움 (~50KB)
단점: 인터넷 연결 필요

오프라인 배포 필요 시 Coqui XTTS-v2 / Kokoro 등으로 백엔드 교체
(synthesize / synthesize_stream 시그니처만 유지하면 server.py 변경 불필요)

- synthesize()         : 동기, 전체 텍스트 → mp3 파일 (legacy)
- synthesize_stream()  : 비동기, mp3 청크별 async generator (used by /voice/ws)
"""
import asyncio
import edge_tts

VOICE = "ko-KR-SunHiNeural"  # 여성. 남성은 "ko-KR-InJoonNeural"


async def _synthesize(text: str, out_path: str, voice: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def synthesize(text: str, out_path: str, voice: str = VOICE) -> str:
    """텍스트 → 음성 파일(mp3) 저장. 저장 경로 반환. (legacy POST /voice 용)"""
    asyncio.run(_synthesize(text, out_path, voice))
    return out_path


async def synthesize_stream(text: str, voice: str = VOICE):
    """
    텍스트 → mp3 chunk async generator.

    edge_tts.Communicate.stream()이 음성 청크 + word boundary 메시지를 섞어 보냄.
    audio 타입만 추출해서 yield.

    Usage:
        async for chunk in synthesize_stream("안녕하세요"):
            await websocket.send_bytes(chunk)
    """
    communicate = edge_tts.Communicate(text, voice)
    async for msg in communicate.stream():
        if msg.get("type") == "audio":
            yield msg["data"]


if __name__ == "__main__":
    # 1) 동기 테스트
    synthesize("안녕하세요. 이력서 작성을 도와드리겠습니다.", "test.mp3")
    print("✅ test.mp3 생성 (동기)")

    # 2) 스트림 테스트
    async def _stream_demo():
        chunks = []
        async for c in synthesize_stream("안녕하세요. 스트리밍 테스트입니다."):
            chunks.append(c)
        with open("test_stream.mp3", "wb") as f:
            f.write(b"".join(chunks))
        print(f"✅ test_stream.mp3 생성 ({len(chunks)} chunks)")

    asyncio.run(_stream_demo())
