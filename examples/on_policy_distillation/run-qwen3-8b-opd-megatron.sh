#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-8b-opd-megatron.sh
#
# OPD with a Megatron-loaded teacher (no external vLLM teacher server).
# Student rollout still uses vLLM. This demo uses the same architecture for
# student and teacher — replace --opd-teacher-load with a stronger checkpoint
# in practice.

set -ex

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

pkill -9 vllm 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3

source "/root/vime/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
    --hf-checkpoint /root/models/Qwen3-8B
    --ref-load /root/models/Qwen3-8B_torch_dist
    --load /root/models/Qwen3-8B_torch_dist
    --save /root/Qwen3-8B_opd/
    --save-interval 20
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl
    --input-key prompt
    --apply-chat-template
    --rollout-shuffle
    --rm-type math
    --num-rollout 300
    --rollout-batch-size 16
    --n-samples-per-prompt 4
    --rollout-max-response-len 16384
    --rollout-temperature 1
    --global-batch-size 64
    --balance-data
)

EVAL_ARGS=(
    # --eval-interval 20
    # --eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl
    # --n-samples-per-eval-prompt 16
    # --eval-max-response-len 16384
    # --eval-top-p 1
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-opd
    --opd-type megatron
    --opd-kl-coef 1.0
    --opd-teacher-load /root/models/Qwen3-8B_torch_dist
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
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
    # --wandb-group qwen3-8b-opd-megatron
    # --wandb-key ${WANDB_KEY}
)

VLLM_ARGS=(
    --rollout-num-gpus-per-engine 1
    --vllm-gpu-memory-utilization 0.4
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --rollout-num-gpus 4 \
    --colocate \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${VLLM_ARGS[@]} \
    ${MISC_ARGS[@]}

pkill -9 vllm 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
