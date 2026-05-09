# RunPod 네트워크 볼륨(/workspace)에 모델·캐시 영구 저장
# setup.sh / load_model.sh / run.sh가 source해서 사용

export WORKSPACE=/workspace

# HuggingFace 모델 캐시 (Whisper large-v3, bge-m3 등)
export HF_HOME="$WORKSPACE/.hf_cache"

# Ollama 모델 저장소 (ollama create 결과)
export OLLAMA_MODELS="$WORKSPACE/.ollama/models"

mkdir -p "$HF_HOME" "$OLLAMA_MODELS"
