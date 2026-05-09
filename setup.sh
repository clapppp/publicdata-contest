#!/usr/bin/env bash
# Silver Voice Resume Agent — 서버 환경 구축
# 대상: PyTorch Ubuntu Docker image / RunPod / RTX 4000 Ada 20GB

set -euo pipefail
source ./env.sh

if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

echo "[1/3] 시스템 패키지 (PyTorch 이미지에 없는 것만)"
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    zstd \
    lshw

echo "[2/4] Python 패키지 설치"
pip install --no-cache-dir -r requirements.txt

echo "[3/4] MeloTTS 설치 (한국어 로컬 TTS, git+pip)"
pip install --no-cache-dir git+https://github.com/myshell-ai/MeloTTS.git

echo "[4/4] Ollama 설치"
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "  이미 설치됨: $(ollama --version 2>&1 | head -1)"
fi

echo ""
echo "환경 구축 완료"
echo "  HF_HOME        = $HF_HOME"
echo "  OLLAMA_MODELS  = $OLLAMA_MODELS"
echo ""
echo "다음 단계:"
echo "  bash load_model.sh    # qwen3 8b GGUF 다운로드 + Ollama 등록"
echo "  bash run.sh           # 서버 실행"
