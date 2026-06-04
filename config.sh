#!/usr/bin/env bash
# config.sh — shared settings and helper functions for all PBS scripts.
# Source after cd'ing to PBS_O_WORKDIR:
#   cd "$PBS_O_WORKDIR"
#   . ./config.sh

MODEL="Qwen/Qwen3.6-35B-A3B"
MODEL_LABEL="qwen36_35b"   # appended to output columns: is_mental_health_qwen36_35b
VLLM_PORT=8000

# Start vLLM in the background.
# Usage:  vllm_start [logfile] [errfile]
#         VLLM_PID=$!          # capture immediately after
vllm_start() {
    local log="${1:-logs/vllm.log}" err="${2:-logs/vllm.err}"
    vllm serve "$MODEL" \
        --port "$VLLM_PORT" \
        --max-model-len 4096 \
        --max-num-batched-tokens 4096 \
        --quantization fp8 \
        --gpu-memory-utilization 0.90 \
        --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
        --override-generation-config '{"enable_thinking": false}' \
        > "$log" 2> "$err" &
}

# Block until vLLM health endpoint responds or timeout expires.
# Usage:  vllm_wait [timeout_seconds]   (default: 1200)
# Returns 1 on timeout.
vllm_wait() {
    local secs=0 limit="${1:-1200}"
    echo "Waiting for vLLM on port $VLLM_PORT..."
    until curl -sf "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
        sleep 10; secs=$((secs + 10))
        if [ $secs -ge $limit ]; then
            echo "ERROR: vLLM did not start within ${limit}s."
            return 1
        fi
    done
    echo "vLLM ready after ${secs}s."
}
