"""DSpark draft model weight export to HF naming for vLLM weight sync.

Adapted from NeMo RL's ``export_eagle_weights_to_hf`` pattern
(``nemo_rl/models/megatron/draft/utils.py:1059``).

The DSpark draft model is a plain ``nn.Module`` (TP=1 in Phase 1) attached as
``policy_chunk.draft_model``. Its parameters already use HF-style naming
(``embed_tokens.weight``, ``layers.0.self_attn.q_proj.weight``, etc.) because
the model is built from ``nn.Linear``/``nn.Embedding``/``nn.RMSNorm`` — the
same modules DeepSpec's HF checkpoint uses.

Export strategy:
    1. Iterate ``draft_model.named_parameters()``
    2. Skip ``embed_tokens`` and ``lm_head`` (shared with policy — exported
       separately from the policy's *current* weights to avoid staleness)
    3. Return ``list[(name, tensor)]`` with plain HF names (no prefix)

For TP=1, no all-gather is needed (params are already full tensors). TP>1
support is deferred to Phase 3+ and will require TP all-gather on the
``q_proj``/``k_proj``/``v_proj``/``o_proj``/``gate_proj``/``up_proj``/``down_proj``
weight matrices.

Naming convention:
    Unlike NeMo RL's Eagle3 (which mixes policy+draft weights in one stream
    with a ``draft.`` prefix and splits on the vLLM side), vime uses vLLM's
    ``start_draft_weight_update`` API to switch the weight transfer engine
    to target the draft model directly. This means draft weights are sent in
    a separate session with plain HF names — no ``draft.`` prefix needed.
    This mirrors vime's existing MTP weight sync pattern.

Shared embed_tokens / lm_head handling:
    DSpark's copies are frozen at init time (``initialize_embeddings_and_head``
    with ``freeze=True``). During RL, the policy's embed/lm_head DO update.
    Exporting DSpark's frozen copies would send stale weights to vLLM's
    drafter. Instead, we export the policy's *current* embed/lm_head so the
    drafter stays in sync with the policy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Parameters that are shared with the policy model (frozen in DSpark).
# These are exported from the policy's current weights, not DSpark's stale copies.
_SHARED_PARAM_NAMES = {"embed_tokens.weight", "lm_head.weight"}


def export_dspark_weights_to_hf(
    draft_model: nn.Module,
    policy_model: nn.Module | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Export DSpark draft model parameters to HF naming (no prefix).

    Names match what vLLM's DSpark drafter ``load_weights`` expects:
    ``embed_tokens.weight``, ``layers.0.self_attn.q_proj.weight``, etc.

    When the policy uses TP>1, Megatron pads the vocab dimension to
    ``make_vocab_size_divisible_by * TP``. vLLM's drafter expects the
    original (unpadded) vocab size, so vocab-dimension weights are
    truncated to ``org_vocab_size`` before export.

    Args:
        draft_model: The DSpark draft model (``policy_chunk.draft_model``).
            Must be the unwrapped model (not DDP-wrapped). Callers should
            unwrap before passing.
        policy_model: The policy model chunk that owns the draft. If provided,
            ``embed_tokens`` and ``lm_head`` are exported from the policy's
            *current* weights (avoiding staleness from DSpark's frozen copies).
            If None, DSpark's frozen copies are exported instead.

    Returns:
        List of ``(name, tensor)`` tuples with plain HF names. Tensors are
        detached data references (callers should clone if they need to mutate).
        Ready for ``_update_bucket_weights_from_distributed``.
    """
    hf_state: list[tuple[str, torch.Tensor]] = []

    # Determine original (unpadded) vocab size for stripping TP padding.
    config = getattr(draft_model, "config", None)
    org_vocab_size = getattr(config, "org_vocab_size", 0) or 0
    padded_vocab_size = getattr(config, "vocab_size", 0) or 0

    def _strip_vocab_padding(name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Truncate vocab dimension if TP padding was applied."""
        if (
            org_vocab_size > 0
            and padded_vocab_size > org_vocab_size
            and tensor.dim() >= 1
            and tensor.shape[0] == padded_vocab_size
        ):
            logger.debug(
                "[DSpark] Stripping vocab padding for %s: %s -> [%d, ...]",
                name,
                tuple(tensor.shape),
                org_vocab_size,
            )
            return tensor[:org_vocab_size].contiguous()
        return tensor

    # 1. Export DSpark-trained parameters (backbone, fc, heads)
    for name, param in draft_model.named_parameters():
        if name in _SHARED_PARAM_NAMES:
            continue
        hf_state.append((name, _strip_vocab_padding(name, param.data)))

    # 2. Export shared embed_tokens / lm_head.
    # embed_tokens: always from policy (RL updates it, DSpark copy is frozen).
    # lm_head: always from draft_model.  For untied pre-trained DSpark
    #   checkpoints (tie_word_embeddings=false), the draft_model's lm_head
    #   is the pre-trained untied weight (std~1.0) which is very different
    #   from the policy's tied lm_head (std~0.02).  Exporting the policy's
    #   lm_head would overwrite the pre-trained one and break inference.
    #   For tied models, draft_model's lm_head is a frozen copy of the
    #   policy's lm_head; vLLM will tie it to embed_tokens anyway.
    if policy_model is not None:
        policy_embed = _get_policy_embedding(policy_model)
        if policy_embed is not None:
            hf_state.append(("embed_tokens.weight", _strip_vocab_padding("embed_tokens.weight", policy_embed.data)))
    else:
        embed = getattr(draft_model, "embed_tokens", None)
        if embed is not None:
            hf_state.append(("embed_tokens.weight", _strip_vocab_padding("embed_tokens.weight", embed.weight.data)))
    lm_head = getattr(draft_model, "lm_head", None)
    if lm_head is not None:
        hf_state.append(("lm_head.weight", _strip_vocab_padding("lm_head.weight", lm_head.weight.data)))

    logger.info(
        "[DSpark] Exported %d draft weight tensors (HF naming) for vLLM sync",
        len(hf_state),
    )
    return hf_state


def _get_policy_embedding(policy_model: nn.Module) -> torch.nn.Parameter | None:
    """Get the policy model's embedding weight.

    Handles Megatron GPTModel's ``embedding`` attribute (VocabParallelEmbedding
    or similar). For TP=1, the weight is already the full tensor.
    """
    embed = getattr(policy_model, "embedding", None)
    if embed is None:
        return None
    # Megatron's VocabParallelEmbedding stores weight as ``weight`` parameter
    weight = getattr(embed, "weight", None)
    if weight is None:
        return None
    return weight


def _get_policy_lm_head(policy_model: nn.Module) -> torch.nn.Parameter | None:
    """Get the policy model's LM head weight.

    Handles Megatron GPTModel's ``output_layer`` attribute, or the shared
    embedding output weight if ``share_embeddings_and_output_weights`` is True.
    """
    # Direct output_layer
    output_layer = getattr(policy_model, "output_layer", None)
    if output_layer is not None:
        weight = getattr(output_layer, "weight", None)
        if weight is not None:
            return weight

    # Shared embedding/output weight (tied weights)
    shared_weight = None
    if hasattr(policy_model, "shared_embedding_or_output_weight"):
        try:
            shared_weight = policy_model.shared_embedding_or_output_weight()
        except Exception:
            shared_weight = None
    if shared_weight is not None:
        return shared_weight

    # Fallback: look for ``lm_head`` attribute
    lm_head = getattr(policy_model, "lm_head", None)
    if lm_head is not None:
        weight = getattr(lm_head, "weight", None)
        if weight is not None:
            return weight

    return None


def _fully_unwrap(chunk: nn.Module) -> nn.Module:
    """Fully unwrap Megatron model: DDP → Float16Module → GPTModel.

    ``chunk.module`` only removes one layer (DDP), leaving Float16Module which
    does NOT have ``draft_model`` — it lives on the GPTModel inside. Use
    ``megatron.core.utils.unwrap_model`` to remove all wrappers.
    """
    from megatron.core.utils import unwrap_model

    try:
        return unwrap_model(chunk)
    except Exception:
        # Fallback: manual unwrap for non-standard wrapping chains
        unwrapped = chunk
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        return unwrapped


def get_dspark_model_from_policy_chunk(
    model_chunks: Sequence[nn.Module],
) -> nn.Module | None:
    """Find the DSpark draft model attached to the last post-process policy chunk.

    The draft model is attached via ``setattr(model, "draft_model", dspark_model)``
    in ``model_provider.py``. It lives on the last chunk (post_process=True,
    last PP stage). This helper fully unwraps Megatron's DDP→Float16Module→GPTModel
    chain to find the ``draft_model`` attribute on the GPTModel.

    Args:
        model_chunks: The sequence of model chunks (``self.model`` in
            ``UpdateWeightFromDistributed`` / ``UpdateWeightFromTensor``).
    Returns:
        The unwrapped DSpark draft model, or None if not attached.
    """
    for chunk in reversed(model_chunks):
        unwrapped = _fully_unwrap(chunk)
        draft_model = getattr(unwrapped, "draft_model", None)
        if draft_model is not None:
            # Unwrap draft's DDP if present
            if hasattr(draft_model, "module"):
                draft_model = draft_model.module
            return draft_model
    return None


def get_policy_chunk_with_draft(
    model_chunks: Sequence[nn.Module],
) -> nn.Module | None:
    """Find the policy chunk that owns the DSpark draft model.

    Returns the fully unwrapped policy chunk (for accessing its embedding/output_layer).
    """
    for chunk in reversed(model_chunks):
        unwrapped = _fully_unwrap(chunk)
        if getattr(unwrapped, "draft_model", None) is not None:
            return unwrapped
    return None


__all__ = [
    "export_dspark_weights_to_hf",
    "get_dspark_model_from_policy_chunk",
    "get_policy_chunk_with_draft",
]
