#!/usr/bin/env bash
# qwen3 14b GGUF (Q4_K_M) HuggingFace 다운로드 → Ollama 등록
# 20GB VRAM 기준 ~8.5GB 사용 (8b 대비 여유 있음)
# OLLAMA_MODELS 가 /workspace 볼륨을 가리키도록 ollama serve 재시작 후 진행

set -euo pipefail
source ./env.sh

REPO_ID="Qwen/Qwen3-14B-GGUF"  # 공식 repo 없으면 후보: bartowski/Qwen3-14B-GGUF
QUANT="q4_k_m"                  # 20GB VRAM에 충분 (8.5GB 사용)
MODEL_DIR="./models"
OLLAMA_NAME="qwen3:14b"

mkdir -p "$MODEL_DIR"

echo "[0/4] 이미 등록된 모델인지 확인"
if ollama show "${OLLAMA_NAME}" &>/dev/null; then
    echo "  ↳ ${OLLAMA_NAME} 이미 등록됨 — 스킵"
    ollama list
    exit 0
fi
echo "  ↳ 미등록 — 다운로드 진행"

echo "[1/4] Ollama 서버 재시작 (OLLAMA_MODELS=$OLLAMA_MODELS 적용)"
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

echo "[2/4] HuggingFace에서 GGUF 다운로드 → $MODEL_DIR"
python - <<PYEOF
from huggingface_hub import list_repo_files, hf_hub_download
import pathlib

repo = "$REPO_ID"
quant = "$QUANT"
out = "$MODEL_DIR"

files = list_repo_files(repo)
matched = sorted(f for f in files if quant in f.lower() and f.endswith(".gguf"))
if not matched:
    raise SystemExit(f"'{quant}' GGUF not found in {repo}. Available: {files}")

target = matched[0]
print(f"  파일: {target}")
hf_hub_download(repo_id=repo, filename=target, local_dir=out)
pathlib.Path("$MODEL_DIR/.gguf_filename").write_text(target)
PYEOF

GGUF_FILE=$(cat "$MODEL_DIR/.gguf_filename")

echo "[3/4] Modelfile 작성 + Ollama 등록"
cat > "$MODEL_DIR/Modelfile" <<EOF
FROM ./$GGUF_FILE

PARAMETER num_ctx 16384
PARAMETER num_gpu 99
EOF
(cd "$MODEL_DIR" && ollama create "$OLLAMA_NAME" -f Modelfile)

echo "[4/4] 원본 GGUF 삭제 (Ollama 블롭으로 이미 복사됨)"
rm -f "$MODEL_DIR/$GGUF_FILE" "$MODEL_DIR/.gguf_filename"

echo ""
echo "완료. 등록된 모델:"
ollama list
