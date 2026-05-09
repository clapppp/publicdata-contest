#!/usr/bin/env bash
# 서버 실행 — env 적용 + Ollama 서버 + FastAPI

set -euo pipefail
source ./env.sh

echo "[1/2] Ollama 서버 (OLLAMA_MODELS=$OLLAMA_MODELS)"
pkill -f "ollama serve" 2>/dev/null || true
sleep 1
nohup ollama serve > /tmp/ollama.log 2>&1 &
for i in {1..15}; do
    sleep 1
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        break
    fi
done
echo "  OK"

echo "[2/2] FastAPI 시작 (HF_HOME=$HF_HOME)"
python server.py
