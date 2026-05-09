"""
tts.py — Microsoft Edge TTS (한국어 Neural Voice)

장점: 한국어 자연스러움 최상, 의존성 가벼움 (~50KB)
단점: 인터넷 연결 필요

오프라인 배포 필요 시 Coqui XTTS-v2 / Kokoro 등으로 백엔드 교체
(synthesize 함수 시그니처만 유지하면 server.py 변경 불필요)
"""
import asyncio
import edge_tts

VOICE = "ko-KR-SunHiNeural"  # 여성. 남성은 "ko-KR-InJoonNeural"


async def _synthesize(text: str, out_path: str, voice: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def synthesize(text: str, out_path: str, voice: str = VOICE) -> str:
    """텍스트 → 음성 파일(mp3) 저장. 저장 경로 반환."""
    asyncio.run(_synthesize(text, out_path, voice))
    return out_path


if __name__ == "__main__":
    synthesize("안녕하세요. 이력서 작성을 도와드리겠습니다.", "test.mp3")
    print("✅ test.mp3 생성")
