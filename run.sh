#!/usr/bin/env bash
# 서버 실행

set -euo pipefail
source ./env.sh

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "❌ ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다."
    echo "   export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

echo "HF_HOME=$HF_HOME"
echo "Claude API 모델: claude-haiku-4-5"
echo ""

python server.py
