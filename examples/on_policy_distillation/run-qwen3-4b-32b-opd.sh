#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-4b-32b-opd.sh
#
# 8×GPU colocate OPD: Qwen3-4B student (GPUs 0-3) + Qwen3-32B vLLM teacher (GPUs 4-7).
# See README.md for checkpoint conversion and data prerequisites.

set -ex

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

VIME_ROOT=/root/vime
NUM_TRAIN_GPUS=4
TEACHER_TP=4
TEACHER_HOST=127.0.0.1
TEACHER_PORT=13141

TEACHER_MODEL_PATH=/root/models/Qwen3-32B
STUDENT_HF=/root/models/Qwen3-4B
STUDENT_TORCH_DIST=/root/models/Qwen3-4B_torch_dist
DATA_DIR=/root/datasets
SAVE_DIR=/root/Qwen3-4B_opd
LOG_DIR=/root/opd_logs

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

source "${VIME_ROOT}/scripts/models/qwen3-4B.sh"

echo "=== Step 1: Clean up previous Ray / training processes ==="
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 -f "train.py" 2>/dev/null || true
sleep 3

echo "=== Step 2: Launch vLLM teacher server (Qwen3-32B, TP=${TEACHER_TP}) ==="
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m vllm.entrypoints.openai.api_server \
    --model "${TEACHER_MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${TEACHER_PORT}" \
    --tensor-parallel-size "${TEACHER_TP}" \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    > "${LOG_DIR}/teacher_vllm.log" 2>&1 &
TEACHER_PID=$!
echo "Teacher vLLM server PID: ${TEACHER_PID}"

echo "Waiting for teacher server to be ready..."
for i in $(seq 1 120); do
    if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
        echo "ERROR: Teacher server process died. Check ${LOG_DIR}/teacher_vllm.log"
        exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${TEACHER_HOST}:${TEACHER_PORT}/health" 2>/dev/null || true)
    if [ "${HTTP_CODE}" = "200" ]; then
        echo "Teacher vLLM server is ready!"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Teacher server failed to start within 10 minutes"
        kill "${TEACHER_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

echo "=== Step 3: Run OPD training (Qwen3-4B student on GPUs 0-3) ==="

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_TRAIN_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

CKPT_ARGS=(
    --hf-checkpoint "${STUDENT_HF}"
    --ref-load "${STUDENT_TORCH_DIST}"
    --load "${STUDENT_TORCH_DIST}"
    --save "${SAVE_DIR}"
    --save-interval 50
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/gsm8k/train.parquet"
    --input-key messages
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type math
    --num-rollout 500
    --rollout-batch-size 32
    --n-samples-per-prompt 4
    --rollout-max-response-len 4096
    --rollout-temperature 0.8
    --global-batch-size 64
)

EVAL_ARGS=(
    --eval-interval 50
    --eval-prompt-data gsm8k "${DATA_DIR}/gsm8k/test.parquet"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 4096
    --eval-top-k 1
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-opd
    --opd-type vllm
    --opd-kl-coef 1.0
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

WANDB_ARGS=(
    # --use-wandb
    # --wandb-project vime-opd
    # --wandb-group qwen3-4b-32b-opd
    # --wandb-key ${WANDB_KEY}
)

VLLM_ARGS=(
    --rollout-num-gpus-per-engine 2
    --vllm-gpu-memory-utilization 0.7
    --vllm-max-num-seqs 32
    --vllm-max-cudagraph-capture-size 16
)

RM_ARGS=(
    --custom-rm-path vime.rollout.on_policy_distillation.reward_func
    --custom-reward-post-process-path vime.rollout.on_policy_distillation.post_process_rewards
    --rm-url "http://${TEACHER_HOST}:${TEACHER_PORT}/inference/v1/generate"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --actor-num-nodes 1
    --actor-num-gpus-per-node "${NUM_TRAIN_GPUS}"
    --colocate
    --make-vocab-size-divisible-by 128
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/vime:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --train-backend megatron \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${VLLM_ARGS[@]} \
    ${RM_ARGS[@]} \
    ${MISC_ARGS[@]} \
    2>&1 | tee "${LOG_DIR}/opd_training.log"

echo "=== Training complete, stopping teacher server ==="
kill "${TEACHER_PID}" 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 -f "train.py" 2>/dev/null || true
echo "=== Done ==="
