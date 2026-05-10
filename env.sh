# RunPod 네트워크 볼륨(/workspace)에 모델·캐시 영구 저장
# setup.sh / run.sh가 source해서 사용

export WORKSPACE=/workspace

# HuggingFace 모델 캐시 (Whisper large-v3, bge-m3 등)
export HF_HOME="$WORKSPACE/.hf_cache"

# Claude API 키 (RunPod 환경변수 또는 직접 설정)
# export ANTHROPIC_API_KEY=sk-ant-...

mkdir -p "$HF_HOME"
