#!/usr/bin/env bash
# qwen3 8b GGUF (Q4_K_M) HuggingFace 다운로드 → Ollama 등록
# OLLAMA_MODELS 가 /workspace 볼륨을 가리키도록 ollama serve 재시작 후 진행

set -euo pipefail
source ./env.sh

REPO_ID="Qwen/Qwen3-8B-GGUF"   # 공식 repo 없으면 후보: bartowski/Qwen3-8B-GGUF
QUANT="q4_k_m"                  # 20GB VRAM에 여유 있어 q5_k_m 도 가능
MODEL_DIR="./models"
OLLAMA_NAME="qwen3:8b"

mkdir -p "$MODEL_DIR"

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

echo "[3/4] Modelfile 작성"
cat > "$MODEL_DIR/Modelfile" <<EOF
FROM ./$GGUF_FILE

PARAMETER num_ctx 8192
PARAMETER num_gpu 99
EOF

echo "[4/4] Ollama 등록 ($OLLAMA_NAME) → $OLLAMA_MODELS 에 저장"
(cd "$MODEL_DIR" && ollama create "$OLLAMA_NAME" -f Modelfile)

echo ""
echo "완료. 등록된 모델:"
ollama list
