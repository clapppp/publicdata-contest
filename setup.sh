#!/usr/bin/env bash
# Silver Voice Resume Agent — 서버 환경 구축
# LLM: Claude API (ANTHROPIC_API_KEY 필요)
# STT: faster-whisper (GPU 또는 CPU)

set -euo pipefail
source ./env.sh

if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

echo "[1/2] 시스템 패키지"
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
    ffmpeg \
    curl

echo "[2/2] Python 패키지 설치"
pip install --no-cache-dir -r requirements.txt


echo ""
echo "환경 구축 완료"
echo "  HF_HOME = $HF_HOME"
echo ""
echo "다음 단계:"
echo "  export ANTHROPIC_API_KEY=sk-ant-...   # Claude API 키 설정"
echo "  bash run.sh                            # 서버 실행"
