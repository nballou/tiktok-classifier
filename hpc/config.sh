#!/usr/bin/env bash
# config.sh — HPC infrastructure: vLLM port and server helpers.
# Model configuration lives in model.env (repo root).
#
# Source order in PBS scripts:
#   . model.env      # MODEL, MODEL_LABEL, QUANTIZATION, LANGUAGE_MODEL_ONLY, DISABLE_THINKING
#   . hpc/config.sh  # VLLM_PORT, vllm_start(), vllm_wait()

VLLM_PORT=8000

# Start vLLM in the background.
# Usage:  vllm_start [logfile] [errfile]
#         VLLM_PID=$!          # capture immediately after
vllm_start() {
    local log="${1:-logs/vllm.log}" err="${2:-logs/vllm.err}"
    local extra_args=()

    [ -n "$QUANTIZATION" ]                     && extra_args+=(--quantization "$QUANTIZATION")
    [ "${LANGUAGE_MODEL_ONLY:-false}" = true ]  && extra_args+=(--language-model-only)
    [ "${DISABLE_THINKING:-false}"    = true ]  && extra_args+=(--default-chat-template-kwargs '{"enable_thinking": false}')

    vllm serve "$MODEL" \
        --port "$VLLM_PORT" \
        --max-model-len 4096 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8]}' \
        --max-num-seqs 64 \
        "${extra_args[@]}" \
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
