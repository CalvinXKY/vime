# On-Policy Distillation Example

This example shows how to run **on-policy distillation (OPD)** with vime. A small
student (Qwen3-4B) learns to match a larger teacher (Qwen3-32B) by training only
on the student's own vLLM rollouts and applying a token-level KL penalty against
the teacher's log-probabilities.

## Key Features

- **OPD is orthogonal to advantage estimators**: OPD adds a KL penalty on top of
  any advantage estimator (GRPO, PPO, REINFORCE++, etc.), not as a separate
  estimator.
- **Two teacher modes**:
  - **`vllm`**: Teacher runs on an external vLLM server; teacher log-probs are
    fetched during rollout via `--rm-url`.
  - **`megatron`**: Teacher is loaded into Megatron via `--opd-teacher-load`;
    teacher log-probs are computed during the training forward pass.
- **Student rollout always uses vLLM** (vime's default rollout backend).

## Files

| File | Description |
|------|-------------|
| `run-qwen3-4b-32b-opd.sh` | 8×GPU colocate demo: Qwen3-4B student + external Qwen3-32B vLLM teacher on GSM8K |
| `run-qwen3-8b-opd-megatron.sh` | Megatron-loaded teacher (no external server); student rollout still uses vLLM |

## GPU Layout (`run-qwen3-4b-32b-opd.sh`)

Single node, 8× GPU:

| GPUs | Role |
|------|------|
| 0–3 | Student Megatron train (TP=2) + student vLLM rollout (TP=2), colocate |
| 4–7 | Teacher vLLM (Qwen3-32B, TP=4) |

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use-opd` | Enable on-policy distillation. |
| `--opd-type` | `vllm` or `megatron`. Required when `--use-opd` is set. |
| `--opd-kl-coef` | OPD KL penalty coefficient (default: 1.0). |
| `--opd-teacher-load` | Teacher checkpoint path. **Required** for `--opd-type=megatron`; must **not** be set for `--opd-type=vllm`. |
| `--rm-url` | Teacher vLLM generate endpoint. Required for `--opd-type=vllm`. |
| `--custom-rm-path` | `vime.rollout.on_policy_distillation.reward_func` |
| `--custom-reward-post-process-path` | `vime.rollout.on_policy_distillation.post_process_rewards` |

## Components

- `vime/rollout/on_policy_distillation.py` implements the vLLM teacher path:
  - `reward_func` POSTs each rollout sample to the teacher vLLM server
    (`--rm-url`) and collects token-level log-probs.
  - `post_process_rewards` trims teacher log-probs to the response span and
    stores them on each `Sample` for the OPD KL term in training.
- Megatron teacher mode computes teacher log-probs inside
  `apply_opd_kl_to_advantages` during the training forward pass.

## OPD Data Flow

```
Student vLLM rollout (GPU 0-3)
  → token sequence + student logprobs
Teacher vLLM (HTTP POST, GPU 4-7)          [vllm mode only]
  → teacher_log_probs per token
post_process_rewards
  → store teacher_log_probs; scalar_rewards=[0.0] (pure distillation)
apply_opd_kl_to_advantages (Megatron)
  → advantages -= opd_kl_coef * (student_logp - teacher_logp)
GRPO policy update
```

## Running the Example

### Prerequisites

```bash
# Models
hf download Qwen/Qwen3-32B --local-dir /root/models/Qwen3-32B
hf download Qwen/Qwen3-4B   --local-dir /root/models/Qwen3-4B

# Data
hf download --repo-type dataset openai/gsm8k --local-dir /root/datasets/gsm8k
# Or use a parquet copy with `messages` + `label` columns under /root/datasets/gsm8k/
```

### Step 1: Convert student checkpoint

Qwen3-4B uses tied embeddings (`tie_word_embeddings=True`) — **do not** pass
`--untie-embeddings-and-output-weights`. Use TP=1 for conversion; training uses
TP=2 at runtime. Pad vocab to 152064 for TP=2 training:

```bash
cd /root/vime
source scripts/models/qwen3-4B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /root/models/Qwen3-4B \
  --save /root/models/Qwen3-4B_torch_dist \
  --padded-vocab-size 152064
```

### Step 2: Run OPD (vLLM teacher)

```bash
cd /root/vime
bash examples/on_policy_distillation/run-qwen3-4b-32b-opd.sh
```

The script will:

1. Launch the Qwen3-32B teacher vLLM server on GPUs 4–7.
2. Start Ray on GPUs 0–3 and submit the OPD training job.
3. Tear down the teacher server and Ray when training finishes.

### Step 3 (optional): Megatron teacher

For same-architecture teacher/student pairs that fit in GPU memory together,
use the Megatron teacher path — no external vLLM teacher server needed:

```bash
# Convert teacher (example uses Qwen3-8B; use a stronger checkpoint in practice)
cd /root/vime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/models/Qwen3-8B \
  --save /root/models/Qwen3-8B_torch_dist

bash examples/on_policy_distillation/run-qwen3-8b-opd-megatron.sh
```

Edit `--opd-teacher-load` in the Megatron script to point at your teacher
checkpoint.

## Preliminary Results

End-to-end run with `run-qwen3-4b-32b-opd.sh` (500 rollout steps, GRPO +
`--opd-kl-coef 1.0`, GSM8K greedy eval, n=1319):

| Model | GSM8K Accuracy |
|-------|----------------|
| Qwen3-4B (pre-OPD) | 78.8% |
| Qwen3-4B (post-OPD, 500 steps) | **85.6%** (+6.8 pp) |
| Qwen3-32B teacher | 88.6% |

Training health signal: `rollout/opd_reverse_kl` dropped from 0.216 → 0.110
(−49%) over 500 steps.

## FAQ

1. **Why two OPD modes?**
   - `vllm`: Teacher on a separate vLLM server. Use when the teacher is larger
     or has a different architecture than the student.
   - `megatron`: Teacher loaded into Megatron. Use when teacher and student share
     architecture and fit in training GPU memory.

2. **Why is `rollout/raw_reward` always 0?**
   Pure OPD distillation does not use an external reward model. The learning
   signal comes entirely from the OPD KL term applied to advantages.

3. **What if I set incompatible arguments?**
   vime validates OPD args at startup:
   - `--use-opd` without `--opd-type` → error
   - `--opd-type megatron` without `--opd-teacher-load` → error
   - `--opd-type vllm` with `--opd-teacher-load` → error

4. **Qwen3-4B checkpoint conversion fails with vocab/TP errors?**
   Re-convert with `--padded-vocab-size 152064` and TP=1 (do not set
   `--tensor-model-parallel-size 2` during conversion). Add
   `--make-vocab-size-divisible-by 128` at training time.

## References

1. https://thinkingmachines.ai/blog/on-policy-distillation/
2. https://arxiv.org/abs/2306.13649
3. https://arxiv.org/abs/2306.08543
