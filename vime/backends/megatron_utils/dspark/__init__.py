"""DSpark (semi-autoregressive speculative decoding) draft model for vime Megatron backend.

This module adapts DeepSpec's DSpark implementation to Megatron-LM for online
draft model training alongside the RL policy. The adaptation follows the
pattern established by NeMo RL's Eagle3 online training:

- DSparkModel is attached as a child attribute of the policy GPTModel chunk
  (``policy_chunk.draft_model``) before DDP wrapping, so gradients and
  optimizer state are managed automatically.
- Hidden states are captured from the policy forward pass via forward hooks
  (``HiddenStateCapture``), then fed to the DSpark draft forward.
- DSpark loss (3-loss: CE + L1/TV + Confidence) is combined with the policy
  RL loss via ``DraftLossWrapper``.
- Draft weights are exported with plain HF names (no ``draft.`` prefix) and
  sent to vLLM via ``start_draft_weight_update`` session mode (same pattern
  as vime's MTP weight sync).

Reference implementations:
- DeepSpec: https://github.com/deepseek-ai/DeepSpec (HF transformers-based)
- NeMo RL Eagle3: https://github.com/NVIDIA/NeMo-RL (Megatron-based pattern)
"""

from .common import (
    AcceptRatePredictor,
    DSparkConfig,
    DSparkForwardOutput,
    build_eval_mask,
    create_dspark_attention_mask,
    create_noise_embed,
    create_position_ids,
    sample_anchor_positions,
)
from .export import export_dspark_weights_to_hf, get_dspark_model_from_policy_chunk, get_policy_chunk_with_draft
from .loss import compute_dspark_loss
from .markov_head import build_markov_head
from .modeling import DSparkModel, build_dspark_model

__all__ = [
    "AcceptRatePredictor",
    "DSparkConfig",
    "DSparkForwardOutput",
    "DSparkModel",
    "build_dspark_model",
    "build_markov_head",
    "build_eval_mask",
    "compute_dspark_loss",
    "create_dspark_attention_mask",
    "create_noise_embed",
    "create_position_ids",
    "export_dspark_weights_to_hf",
    "get_dspark_model_from_policy_chunk",
    "get_policy_chunk_with_draft",
    "sample_anchor_positions",
]
