"""
Qwen3-Omni-MoE (qwen3_omni_moe) bridge for megatron.bridge.

Registers `Qwen3OmniMoeForConditionalGeneration` so that
`AutoBridge.from_hf_pretrained` recognises Qwen3-Omni-30B-A3B checkpoints and
can provide a Megatron-compatible Thinker model + weight mappings.

Architecture (Thinker-only, Talker/Code2Wav are frozen and not trained):
  HF audio encoder  (Qwen3OmniMoeAudioEncoder,  replicated on first PP stage)
  HF vision encoder (Qwen3OmniMoeVisionEncoder, replicated on first PP stage)
  + Megatron GPTModel (MoE language model with M-RoPE, deepstack)

The forward pass:
  1. Computes text embeddings from `input_ids`.
  2. Runs the HF vision encoder on `pixel_values`+`image_grid_thw`
     (and `pixel_values_videos`+`video_grid_thw` if present), scatters the
     resulting vision embeddings into the combined embedding tensor at
     positions where `input_ids == image_token_id` / `video_token_id`.
  3. Runs the HF audio encoder on `input_features`+`feature_attention_mask`,
     scatters the resulting audio embeddings at positions where
     `input_ids == audio_token_id`.
  4. Computes 3D M-RoPE position IDs from the full input_ids + grid info
     (audio-aware, ported from Relax's get_rope_index).
  5. Forwards the combined embeddings + M-RoPE position IDs through the
     Megatron GPTModel (MoE) language model.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping, ReplicatedMapping
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import Qwen3VLMultimodalRotaryEmbedding
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import Qwen3VLTransformerBlock
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import split_deepstack_embs
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from megatron.core import InferenceParams, mpu, parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import deprecate_inference_params

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Register Qwen3OmniMoe with AutoModelForCausalLM so that
# `megatron.bridge.AutoBridge.from_hf_pretrained` (which uses
# `PreTrainedCausalLM._load_model` -> `AutoModelForCausalLM.from_pretrained`)
# can load the multimodal `Qwen3OmniMoeForConditionalGeneration` model.
# Without this, AutoModelForCausalLM rejects `Qwen3OmniMoeConfig` because
# multimodal conditional-generation models are not registered as CausalLMs.
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModelForCausalLM
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration

    if "qwen3_omni_moe" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["qwen3_omni_moe"] = "Qwen3OmniMoeForConditionalGeneration"
        # Also register with the AutoModelForCausalLM class (for mappings built before this point)
        try:
            AutoModelForCausalLM.register(Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration)
        except ValueError:
            # Already registered or mapping conflict; ignore
            pass
except ImportError:
    pass


# ---------------------------------------------------------------------------
# THD <-> batch-sequence helpers (cf. Qwen3VL bridge / glm4v_moe)
# ---------------------------------------------------------------------------
def _thd_to_batch_seq(packed: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Unpack THD-format [1, T, ...] to [bs, max_seq, ...] using cu_seqlens."""
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq = seqlens.max().item()
    bs = len(cu_seqlens) - 1
    out = packed.new_zeros(bs, max_seq, *packed.shape[2:])
    for i, sl in enumerate(seqlens):
        out[i, :sl] = packed[0, cu_seqlens[i] : cu_seqlens[i] + sl]
    return out


def _batch_seq_to_thd(unpacked: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Pack [bs, max_seq, ...] back to THD [1, T, ...]."""
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    total = cu_seqlens[-1].item()
    out = unpacked.new_zeros(1, total, *unpacked.shape[2:])
    for i, sl in enumerate(seqlens):
        out[0, cu_seqlens[i] : cu_seqlens[i] + sl] = unpacked[i, :sl]
    return out


def _gather_input_ids_from_cp(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct full (global) input_ids across CP ranks.

    Approximate: concatenates per-rank chunks in rank order. That matches
    non-zigzag CP; zigzag CP would need the [r] / [2*cp-1-r] reassembly.
    Current Omni VLM training keeps CP=1 or non-zigzag, so this is enough.
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return input_ids

    cp_group = parallel_state.get_context_parallel_group()

    # Each rank's local input_ids correspond to its cp chunk of each sequence
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    global_seqlens = [None] * cp_size
    torch.distributed.all_gather_object(global_seqlens, seqlens, group=cp_group)

    # For each sequence, gather the full token range across CP ranks
    bs = len(seqlens)
    gathered_per_rank: list[torch.Tensor] = [None] * cp_size
    torch.distributed.all_gather_object(gathered_per_rank, input_ids, group=cp_group)

    out_chunks = []
    for i in range(bs):
        # Non-zigzag: concatenate ranks in order (see docstring).
        seq_parts = []
        for r in range(cp_size):
            local_sl = global_seqlens[r][i]
            # Extract this sequence's local portion from rank r
            # This is approximate; for full correctness, use the CP-aware
            # implementation from Megatron's transformer.py
            offset = sum(global_seqlens[r][:i])
            seq_parts.append(gathered_per_rank[r][0, offset : offset + local_sl].to(input_ids.device))
        out_chunks.append(torch.cat(seq_parts, dim=0))

    max_seq = max(c.numel() for c in out_chunks)
    out = input_ids.new_zeros(1, max_seq * bs)
    for i, c in enumerate(out_chunks):
        out[0, i * max_seq : i * max_seq + c.numel()] = c
    return out


# ---------------------------------------------------------------------------
# Audio-aware M-RoPE position ID computation (ported from Relax)
# ---------------------------------------------------------------------------
def _get_feat_extract_output_lengths(input_lengths):
    """Computes the output length of the conv layers and the audio encoder."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


def _get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    audio_token_id: int,
    vision_start_token_id: int,
    audio_start_token_id: int,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    audio_seqlens: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    use_audio_in_video: bool = False,
    second_per_grids: torch.Tensor | None = None,
    position_id_per_seconds: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate RoPE position indices for multimodal inputs (audio+image+video).

    Ported from relax.models.qwen_omni.modeling_qwen3_omni.utils.get_rope_index.
    Returns position_ids of shape [3, batch, seq] for M-RoPE.
    """
    # Do NOT split video_grid_thw by repeat_interleave.
    # The Qwen3-Omni processor does NOT insert timestamp tokens between video
    # frames for pure video (use_audio_in_video=False). Input_ids have ONE
    # <vision_start> + (grid_t*grid_h*grid_w/merge^2) <video_pad> + <vision_end>.
    # Splitting grid_t into t=1 entries would under-count video tokens and break
    # M-RoPE vs vLLM (see vLLM get_mrope_input_positions).

    mrope_position_deltas = []
    if image_grid_thw is not None or video_grid_thw is not None or audio_seqlens is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=torch.float,
            device=input_ids.device,
        )
        image_index, video_index, audio_index = 0, 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids_i in enumerate(total_input_ids):
            input_ids_i = input_ids_i[attention_mask[i] == 1]

            vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids_i[vision_start_indices + 1]
            audio_nums = torch.sum(input_ids_i == audio_start_token_id)
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (
                (vision_tokens == audio_start_token_id).sum()
                if use_audio_in_video
                else (vision_tokens == video_token_id).sum()
            )
            input_tokens = input_ids_i.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
            multimodal_nums = image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums

            for _ in range(multimodal_nums):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                    remain_videos > 0 or remain_images > 0
                ):
                    ed_vision_start = input_tokens.index(vision_start_token_id, st)
                else:
                    ed_vision_start = len(input_tokens) + 1
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio_start = input_tokens.index(audio_start_token_id, st)
                else:
                    ed_audio_start = len(input_tokens) + 1
                min_ed = min(ed_vision_start, ed_audio_start)

                # ---------- text ----------
                text_len = min_ed - st
                if text_len > 0:
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    st_idx += text_len

                # ---------- BOS ----------
                if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                    bos_len, eos_len = 2, 2
                else:
                    bos_len, eos_len = 1, 1

                llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                st_idx += bos_len

                # Audio Only
                if min_ed == ed_audio_start:
                    audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_index])
                    llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    llm_pos_ids_list.append(llm_pos_ids)

                    st += text_len + bos_len + audio_len + eos_len
                    audio_index += 1
                    remain_audios -= 1

                # Image Only
                elif min_ed == ed_vision_start and input_ids_i[ed_vision_start + 1] == image_token_id:
                    t, h, w = (
                        image_grid_thw[image_index][0].item(),
                        image_grid_thw[image_index][1].item(),
                        image_grid_thw[image_index][2].item(),
                    )
                    t_index = (torch.arange(t) * 1 * position_id_per_seconds).float()
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
                    )
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
                    )
                    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
                    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
                    llm_pos_ids_list.append(_llm_pos_ids + st_idx)

                    image_len = image_grid_thw[image_index].prod().item() // (spatial_merge_size**2)
                    st += int(text_len + bos_len + image_len + eos_len)
                    image_index += 1
                    remain_images -= 1

                # Video Only
                elif min_ed == ed_vision_start and input_ids_i[ed_vision_start + 1] == video_token_id:
                    t, h, w = (
                        video_grid_thw[video_index][0].item(),
                        video_grid_thw[video_index][1].item(),
                        video_grid_thw[video_index][2].item(),
                    )
                    t_index = (
                        torch.arange(t) * second_per_grids[video_index].cpu().float() * position_id_per_seconds
                    ).float()
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
                    )
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
                    )
                    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
                    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
                    llm_pos_ids_list.append(_llm_pos_ids + st_idx)

                    video_len = video_grid_thw[video_index].prod().item() // (spatial_merge_size**2)
                    st += int(text_len + bos_len + video_len + eos_len)
                    video_index += 1
                    remain_videos -= 1

                # Audio in Video
                elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                    audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_index])
                    audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx

                    t, h, w = (
                        video_grid_thw[video_index][0].item(),
                        video_grid_thw[video_index][1].item(),
                        video_grid_thw[video_index][2].item(),
                    )
                    t_index = (
                        torch.arange(t) * second_per_grids[video_index].cpu().float() * position_id_per_seconds
                    ).float()
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
                    )
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
                    )
                    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
                    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
                    llm_pos_ids_list_temp = [_llm_pos_ids + st_idx]
                    video_llm_pos_ids = torch.cat(llm_pos_ids_list_temp, dim=1)

                    video_data_index, audio_data_index = 0, 0
                    while (
                        video_data_index < video_llm_pos_ids.shape[-1]
                        and audio_data_index < audio_llm_pos_ids.shape[-1]
                    ):
                        if video_llm_pos_ids[0][video_data_index] <= audio_llm_pos_ids[0][audio_data_index]:
                            llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
                            video_data_index += 1
                        else:
                            llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
                            audio_data_index += 1
                    if video_data_index < video_llm_pos_ids.shape[-1]:
                        llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]])
                    if audio_data_index < audio_llm_pos_ids.shape[-1]:
                        llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]])
                    video_len = video_grid_thw[video_index].prod().item() // (spatial_merge_size**2)

                    st += int(text_len + bos_len + audio_len + video_len + eos_len)
                    audio_index += 1
                    video_index += 1
                    remain_videos -= 1
                    remain_audios -= 1
                else:
                    raise RuntimeError("unexpected error in get_rope_index")

                # ---------- EOS ----------
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

            # tail text
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids_i))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        # fallback (pure text)
        if attention_mask is not None:
            position_ids = attention_mask.float().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


# ---------------------------------------------------------------------------
# GPTModel with DeepStack support (keeps MCore mrope, swaps decoder only)
# ---------------------------------------------------------------------------
class Qwen3OmniMoeGPTModel(MCoreGPTModel):
    """Qwen3-Omni GPT model with DeepStack support.

    Inherits GPTModel to keep MCore's MultimodalRotaryEmbedding (proven for text
    training). Only replaces decoder with Qwen3VLTransformerBlock to add DeepStack
    injection at the first N decoder layers.
    """

    def __init__(
        self,
        config,
        transformer_layer_spec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: str = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor=None,
        mtp_block_spec=None,
        vp_stage=None,
        pg_collection=None,
    ) -> None:
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )
        # Rebuild rotary_pos_emb with Qwen3VLMultimodalRotaryEmbedding.
        # CRITICAL: MCore's MultimodalRotaryEmbedding uses NON-interleaved mrope
        # layout [T48,H40,W40,...] which diverges from vLLM/HF interleaved layout
        # [T24,H24,W24,...] when t!=h!=w (video). For text (t=h=w) both are identical.
        # Bridge's Qwen3VLMultimodalRotaryEmbedding.apply_interleaved_mrope matches
        # HF/vLLM exactly -> fixes video logprob_abs_diff=1.77.
        cp_group = None
        if pg_collection is not None and getattr(pg_collection, "cp", None) is not None:
            cp_group = pg_collection.cp
        else:
            from megatron.core import parallel_state

            cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
        self.rotary_pos_emb = Qwen3VLMultimodalRotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=False,  # bridge asserts not interleaved; uses apply_interleaved_mrope internally
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            cp_group=cp_group,
        )
        # Rebuild decoder as Qwen3VLTransformerBlock (adds DeepStack injection).
        self.decoder = Qwen3VLTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
        runtime_gather_output=None,
        *,
        inference_params=None,
        loss_mask=None,
        # args for deepstack
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
    ):
        """Forward pass with DeepStack visual embedding injection."""
        inference_context = deprecate_inference_params(inference_context, inference_params)

        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )
        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ) = preproc_output[:5]

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
        )

        result = self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )
        return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Qwen3OmniMoeVLModel(MegatronModule):
    """Qwen3-Omni-MoE Thinker model for Megatron training.

    Wraps an HF audio encoder and an HF vision encoder (only on first PP stage)
    together with a standard Megatron Core GPTModel configured for M-RoPE
    (MoE language model).

    Thinker-only training: the Talker and Code2Wav modules are not loaded.
    The audio and vision encoders are frozen by default (RL only trains the
    language model).
    """

    # Shared across instances so call_N.pt dumps are not double-consumed.
    _used_call_indices: set | None = None

    @classmethod
    def _get_used_call_indices(cls, features_dir: str) -> set:
        """Return the shared used-index set, pruning entries whose dumps vanished.

        Clearing the dump directory (or starting a new run without restarting the
        process) removes call_*.pt files; drop those indices so new dumps can be
        matched again.
        """
        if cls._used_call_indices is None:
            cls._used_call_indices = set()
        existing = set()
        try:
            for name in os.listdir(features_dir):
                if name.startswith("call_") and name.endswith(".pt"):
                    try:
                        existing.add(int(name[len("call_") : -len(".pt")]))
                    except ValueError:
                        continue
        except OSError:
            existing = set()
        cls._used_call_indices = {i for i in cls._used_call_indices if i in existing}
        return cls._used_call_indices

    @staticmethod
    def _cleanup_feature_dumps(features_dir: str, prefix: str) -> None:
        """Bound dump-file growth under ``features_dir``.

        Keeps the newest ``VIME_OMNI_FEATURES_MAX_FILES`` dumps (default 256).
        Set the env to ``0`` to disable cleanup. Matched files may still linger
        until the next cleanup pass.
        """
        try:
            max_files = int(os.environ.get("VIME_OMNI_FEATURES_MAX_FILES", "256"))
        except ValueError:
            max_files = 256
        if max_files <= 0:
            return
        try:
            paths = [
                os.path.join(features_dir, name)
                for name in os.listdir(features_dir)
                if name.startswith(prefix) and name.endswith(".pt")
            ]
        except OSError:
            return
        if len(paths) <= max_files:
            return
        paths.sort(key=lambda p: os.path.getmtime(p))
        for path in paths[: len(paths) - max_files]:
            try:
                os.unlink(path)
            except OSError:
                pass

    def __init__(
        self,
        language_transformer_config,
        language_transformer_layer_spec,
        hf_audio_config,
        hf_vision_config,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection=None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.pg_collection = pg_collection

        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.vision_start_token_id = language_transformer_config.vision_start_token_id
        self.audio_token_id = language_transformer_config.audio_token_id
        self.audio_start_token_id = language_transformer_config.audio_start_token_id
        self.spatial_merge_size = language_transformer_config.spatial_merge_size
        self.position_id_per_seconds = language_transformer_config.position_id_per_seconds
        self.use_audio_in_video = getattr(language_transformer_config, "use_audio_in_video", False)

        self.share_embeddings_and_output_weights = False

        # Encoders -- only on the first pipeline stage
        self.audio_model = None
        self.vision_model = None
        if self.pre_process:
            from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
                Qwen3OmniMoeAudioEncoder,
                Qwen3OmniMoeVisionEncoder,
            )

            self.audio_model = Qwen3OmniMoeAudioEncoder._from_config(hf_audio_config)
            self.vision_model = Qwen3OmniMoeVisionEncoder._from_config(hf_vision_config)
            # Freeze encoders -- not trained during RL
            self.audio_model.requires_grad_(False)
            self.audio_model.eval()
            self.vision_model.requires_grad_(False)
            self.vision_model.eval()

            # Ensure HF encoder params are marked for TP grad sync and future assignments are hooked.
            hook_hf_module_setattr_for_tp_grad_sync(self.audio_model)
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)
            if torch.cuda.is_available():
                # Keep encoder param dtype (often bf16 from HF config); only move device.
                _enc_device = torch.device(f"cuda:{torch.cuda.current_device()}")
                _audio_dtype = next(self.audio_model.parameters()).dtype
                _vision_dtype = next(self.vision_model.parameters()).dtype
                self.audio_model = self.audio_model.to(device=_enc_device, dtype=_audio_dtype)
                self.vision_model = self.vision_model.to(device=_enc_device, dtype=_vision_dtype)

        # Cache for vLLM encoder feature dumps (fingerprint -> tensors).
        self._vllm_features_cache = {}
        self._vllm_audio_cache = {}
        # Language model -- Megatron GPT with M-RoPE + DeepStack support
        self.language_model = Qwen3OmniMoeGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
            pg_collection=pg_collection,
        )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    # -- helpers required by Megatron pipeline engine -----------------------

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    # -- encoder helpers ----------------------------------------------------

    def _get_vision_features(self, pixel_values, image_grid_thw):
        """Use vLLM vision feature dumps when available, else HF encoder."""
        vllm_features_dir = os.environ.get("VIME_OMNI_VISION_FEATURES_DIR", "/tmp/vime_omni_vision_features")
        use_vllm = os.path.exists(vllm_features_dir)

        if use_vllm:
            from megatron.core import parallel_state as mpu

            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_group = mpu.get_tensor_model_parallel_group()
            tp_src = mpu.get_tensor_model_parallel_src_rank()
            # Match dumps by grid_thw + pixel fingerprint
            vllm_main = None
            vllm_ds = None
            matched_call_idx = None
            if tp_rank == 0:
                Qwen3OmniMoeVLModel._cleanup_feature_dumps(vllm_features_dir, "call_")
                # Handle batch grid_thw by matching per-video
                target_grid_thw_batch = (
                    image_grid_thw.tolist() if hasattr(image_grid_thw, "tolist") else image_grid_thw
                )
                num_videos = len(target_grid_thw_batch)
                target_fingerprint = pixel_values.float().sum().item()
                used_call_indices = Qwen3OmniMoeVLModel._get_used_call_indices(vllm_features_dir)
                # Clear step-local cache each call to avoid cross-step false hits.
                self._step_local_cache = {}

                # Try to match the full batch first (for single-video batches)
                cache_key = (str(target_grid_thw_batch), round(target_fingerprint, -4))
                if cache_key in self._vllm_features_cache:
                    vllm_main, vllm_ds = self._vllm_features_cache[cache_key]
                    matched_call_idx = -1
                else:
                    import glob as _glob

                    # For multi-video batches, match each video separately
                    # Use per-video cache to handle deduplicated vLLM requests
                    all_matched = []
                    # Per-video fingerprint matching against vLLM dumps.

                    # Calculate per-video patch offsets for fingerprint computation.
                    # Qwen VL packs pixel_values as [sum(t*h*w), patch_dim] in the same
                    # row order as grid_thw (HF processor / vLLM multimodal). Each dump
                    # is typically one image/video; we slice the packed tensor the same way.
                    _offset = 0
                    _video_offsets = []
                    for vi in range(num_videos):
                        _g = target_grid_thw_batch[vi]
                        _grid_prod = int(_g[0]) * int(_g[1]) * int(_g[2])
                        _video_offsets.append((_offset, _grid_prod))
                        _offset += _grid_prod
                    if _offset != pixel_values.shape[0]:
                        logger.warning(
                            "Vision grid_thw patch sum (%s) != pixel_values rows (%s); "
                            "per-video fingerprint offsets may be wrong",
                            _offset,
                            pixel_values.shape[0],
                        )

                    for video_idx in range(num_videos):
                        target_grid_thw_single = [target_grid_thw_batch[video_idx]]
                        # Per-video fingerprint on the same raw pixel_values as vLLM dumps.
                        _v_off, _v_prod = _video_offsets[video_idx]
                        per_video_fp = pixel_values[_v_off : _v_off + _v_prod].float().sum().item()
                        # Cache key includes fingerprint to distinguish same-grid_thw videos
                        step_cache_key = (str(target_grid_thw_single), round(per_video_fp, -4))

                        # Check step-local cache first (handles deduplicated videos within same step)
                        if step_cache_key in self._step_local_cache:
                            vllm_main_single, vllm_ds_single = self._step_local_cache[step_cache_key]
                            all_matched.append((vllm_main_single, vllm_ds_single, -1))
                            continue

                        # Not in cache, scan files in call_idx order
                        # Ensure each dump file is consumed at most once.
                        _call_files = sorted(
                            _glob.glob(f"{vllm_features_dir}/call_*.pt"),
                            key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]),
                        )
                        for call_file in _call_files:
                            call_idx = int(os.path.basename(call_file).split("_")[1].split(".")[0])
                            if call_idx in used_call_indices:
                                continue
                            try:
                                saved = torch.load(call_file, map_location="cpu", weights_only=True)
                                saved_grid_thw = saved.get("grid_thw", None)
                                if saved_grid_thw is not None:
                                    saved_grid_thw_list = (
                                        saved_grid_thw.tolist()
                                        if hasattr(saved_grid_thw, "tolist")
                                        else saved_grid_thw
                                    )
                                    if saved_grid_thw_list == target_grid_thw_single:
                                        saved_fp = saved.get("pixel_fingerprint", None)
                                        if saved_fp is not None:
                                            fp_diff = abs(saved_fp - per_video_fp)
                                            fp_tol = abs(per_video_fp) * 0.02 + 1.0
                                            if fp_diff > fp_tol:
                                                continue
                                        vllm_main_single = saved.get("main_output", None)
                                        vllm_ds_single = saved.get("deepstack_features", None)
                                        matched_call_idx = call_idx
                                        used_call_indices.add(call_idx)
                                        self._step_local_cache[step_cache_key] = (vllm_main_single, vllm_ds_single)
                                        all_matched.append((vllm_main_single, vllm_ds_single, matched_call_idx))
                                        break
                            except Exception as e:
                                logger.warning(
                                    "Failed to load vLLM vision dump %s: %s",
                                    call_file,
                                    e,
                                )
                                continue

                    if len(all_matched) == num_videos:
                        # Concatenate all matched features
                        vllm_main = torch.cat([m[0] for m in all_matched], dim=0)
                        if all_matched[0][1] is not None:
                            vllm_ds = [
                                torch.cat([m[1][i] for m in all_matched], dim=0) for i in range(len(all_matched[0][1]))
                            ]
                        else:
                            vllm_ds = None
                        self._vllm_features_cache[cache_key] = (vllm_main, vllm_ds)
                    else:
                        vllm_main = None
                        vllm_ds = None

            # Broadcast whether features were loaded (1=yes, 0=no)
            # All tensors must be on CUDA for NCCL broadcast
            _cuda_dev = pixel_values.device
            # Move rank-0 loaded tensors from CPU to CUDA before broadcast
            if vllm_main is not None:
                vllm_main = vllm_main.to(device=_cuda_dev, dtype=self.vision_model.dtype)
            if vllm_ds is not None:
                vllm_ds = [f.to(device=_cuda_dev, dtype=self.vision_model.dtype) for f in vllm_ds]

            has_features = torch.tensor([1 if vllm_main is not None else 0], dtype=torch.long, device=_cuda_dev)
            torch.distributed.broadcast(has_features, src=tp_src, group=tp_group)

            if has_features.item():
                # Broadcast main output shape then data
                if tp_rank == 0:
                    shape_len = torch.tensor([len(vllm_main.shape)], dtype=torch.long, device=_cuda_dev)
                    main_shape = torch.tensor(vllm_main.shape, dtype=torch.long, device=_cuda_dev)
                else:
                    shape_len = torch.tensor([0], dtype=torch.long, device=_cuda_dev)
                torch.distributed.broadcast(shape_len, src=tp_src, group=tp_group)
                if tp_rank != 0:
                    main_shape = torch.zeros(shape_len.item(), dtype=torch.long, device=_cuda_dev)
                torch.distributed.broadcast(main_shape, src=tp_src, group=tp_group)
                if tp_rank != 0:
                    vllm_main = torch.zeros(*main_shape.tolist(), dtype=self.vision_model.dtype, device=_cuda_dev)
                torch.distributed.broadcast(vllm_main, src=tp_src, group=tp_group)

                # Broadcast deepstack features — each layer can differ from main_output shape.
                n_ds = torch.tensor([len(vllm_ds) if vllm_ds else 0], dtype=torch.long, device=_cuda_dev)
                torch.distributed.broadcast(n_ds, src=tp_src, group=tp_group)
                if n_ds.item() > 0:
                    if tp_rank == 0:
                        ds_shapes = [torch.tensor(f.shape, dtype=torch.long, device=_cuda_dev) for f in vllm_ds]
                        ds_ndim = torch.tensor([len(f.shape) for f in vllm_ds], dtype=torch.long, device=_cuda_dev)
                    else:
                        ds_ndim = torch.zeros(n_ds.item(), dtype=torch.long, device=_cuda_dev)
                    torch.distributed.broadcast(ds_ndim, src=tp_src, group=tp_group)
                    if tp_rank != 0:
                        ds_shapes = [
                            torch.zeros(int(ds_ndim[i].item()), dtype=torch.long, device=_cuda_dev)
                            for i in range(n_ds.item())
                        ]
                    for i in range(n_ds.item()):
                        torch.distributed.broadcast(ds_shapes[i], src=tp_src, group=tp_group)
                    if tp_rank != 0:
                        vllm_ds = [
                            torch.zeros(*ds_shapes[i].tolist(), dtype=self.vision_model.dtype, device=_cuda_dev)
                            for i in range(n_ds.item())
                        ]
                    for f in vllm_ds:
                        torch.distributed.broadcast(f, src=tp_src, group=tp_group)

                vision_embeds = vllm_main
                deepstack_features = vllm_ds
                return vision_embeds, deepstack_features
            else:
                # Fail fast: silent HF fallback would mix vLLM and HF features across ranks.
                _rank_info = f"global_rank={torch.distributed.get_rank()}, tp_rank={tp_rank}"
                raise RuntimeError(
                    f"[VIME-OMNI] FATAL: No vLLM vision features matched (flag=0). "
                    f"Silent HF fallback would mix vLLM and HF features. "
                    f"Rank info: {_rank_info}. "
                    f"Check: 1) VIME_OMNI_VISION_FEATURES_DIR has call_*.pt files, "
                    f"2) grid_thw matches between vLLM and bridge, "
                    f"3) vLLM rollout completed before training step."
                )

        # Original HF vision encoder path (fallback)
        pixel_values = pixel_values.to(dtype=self.vision_model.dtype)
        with torch.no_grad():
            outputs = self.vision_model(pixel_values, grid_thw=image_grid_thw)

        import transformers
        from packaging import version

        if version.parse(transformers.__version__) >= version.parse("5.0.0"):
            vision_embeds = outputs.pooler_output
            deepstack_features = outputs.deepstack_features
        else:
            vision_embeds, deepstack_features = outputs

        return vision_embeds, deepstack_features

    def _get_audio_features(self, input_features, feature_lens):
        """Use vLLM audio feature dumps when available, else HF encoder."""
        vllm_audio_dir = os.environ.get("VIME_OMNI_AUDIO_FEATURES_DIR", "/tmp/vime_omni_audio_features")
        use_vllm = os.path.exists(vllm_audio_dir)

        if use_vllm:
            from megatron.core import parallel_state as mpu

            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_group = mpu.get_tensor_model_parallel_group()
            tp_src = mpu.get_tensor_model_parallel_src_rank()

            vllm_audio = None
            if tp_rank == 0:
                Qwen3OmniMoeVLModel._cleanup_feature_dumps(vllm_audio_dir, "audio_")
                target_feature_lens = feature_lens.tolist() if hasattr(feature_lens, "tolist") else feature_lens
                num_samples = len(target_feature_lens)

                # Per-sample fingerprints on mask-flattened features.
                # Layout from forward(): [mel_bins, total_time] (time packed on dim 1).
                # Slice time with [:, offset:offset+fl]; do NOT treat dim 0 as batch.
                if input_features.dim() != 2:
                    raise ValueError(
                        f"Expected flattened audio features [mel_bins, total_time], "
                        f"got shape {tuple(input_features.shape)}"
                    )
                per_sample_fps = []
                offset = 0
                for fl in target_feature_lens:
                    fl_int = int(fl)
                    sample_fp = input_features[:, offset : offset + fl_int].float().sum().item()
                    per_sample_fps.append(sample_fp)
                    offset += fl_int
                if offset != input_features.shape[1]:
                    logger.warning(
                        "Audio feature_lens sum (%s) != flattened time dim (%s); fingerprints may mismatch",
                        offset,
                        input_features.shape[1],
                    )

                # Cache by per-sample fingerprint tuple
                cache_key = str([round(fp, -3) for fp in per_sample_fps])
                if cache_key in self._vllm_audio_cache:
                    vllm_audio = self._vllm_audio_cache[cache_key]
                else:
                    import glob as _glob

                    all_files = sorted(_glob.glob(f"{vllm_audio_dir}/audio_*.pt"))

                    # Load all files and index by fingerprint
                    file_index = {}  # fingerprint_rounded -> list of (file, audio_output)
                    for call_file in all_files:
                        try:
                            saved = torch.load(call_file, map_location="cpu", weights_only=True)
                            saved_fp = saved.get("input_fingerprint", None)
                            saved_audio = saved.get("audio_output", None)
                            if saved_fp is not None and saved_audio is not None:
                                fp_key = round(saved_fp, -3)
                                if fp_key not in file_index:
                                    file_index[fp_key] = []
                                file_index[fp_key].append((call_file, saved_audio))
                        except Exception as e:
                            logger.warning(
                                "Failed to load vLLM audio dump %s: %s",
                                call_file,
                                e,
                            )
                            continue

                    # Match each sample by fingerprint (2% tolerance for bf16 vs float32)
                    audio_outputs = []
                    matched_names = []
                    all_matched = True
                    for target_fp in per_sample_fps:
                        target_key = round(target_fp, -3)
                        best_match = None
                        best_diff = float("inf")
                        # Search nearby keys within tolerance
                        for fp_key, candidates in file_index.items():
                            if len(candidates) == 0:
                                continue
                            diff = abs(fp_key - target_key)
                            if diff < abs(target_fp) * 0.02 and diff < best_diff:
                                best_diff = diff
                                best_match = candidates[0]  # Take first available

                        if best_match is not None:
                            call_file, audio_output = best_match
                            audio_outputs.append(audio_output)
                            matched_names.append(os.path.basename(call_file))
                            # Keep dump entries: identical fingerprints share the same features.
                        else:
                            all_matched = False

                    if all_matched and len(audio_outputs) == num_samples:
                        vllm_audio = torch.cat(audio_outputs, dim=0)
                        self._vllm_audio_cache[cache_key] = vllm_audio

            # Broadcast whether features were loaded
            _cuda_dev = input_features.device
            if vllm_audio is not None:
                vllm_audio = vllm_audio.to(device=_cuda_dev, dtype=self.audio_model.dtype)

            has_features = torch.tensor([1 if vllm_audio is not None else 0], dtype=torch.long, device=_cuda_dev)
            torch.distributed.broadcast(has_features, src=tp_src, group=tp_group)

            if has_features.item():
                # Broadcast shape then data
                if tp_rank == 0:
                    shape_len = torch.tensor([len(vllm_audio.shape)], dtype=torch.long, device=_cuda_dev)
                    audio_shape = torch.tensor(vllm_audio.shape, dtype=torch.long, device=_cuda_dev)
                else:
                    shape_len = torch.tensor([0], dtype=torch.long, device=_cuda_dev)
                torch.distributed.broadcast(shape_len, src=tp_src, group=tp_group)
                if tp_rank != 0:
                    audio_shape = torch.zeros(shape_len.item(), dtype=torch.long, device=_cuda_dev)
                torch.distributed.broadcast(audio_shape, src=tp_src, group=tp_group)
                if tp_rank != 0:
                    vllm_audio = torch.zeros(*audio_shape.tolist(), dtype=self.audio_model.dtype, device=_cuda_dev)
                torch.distributed.broadcast(vllm_audio, src=tp_src, group=tp_group)

                return vllm_audio
            else:
                _rank_info = f"global_rank={torch.distributed.get_rank()}, tp_rank={tp_rank}"
                raise RuntimeError(
                    f"[VIME-OMNI] FATAL: No vLLM audio features matched (flag=0). "
                    f"Rank info: {_rank_info}. "
                    f"Check: 1) VIME_OMNI_AUDIO_FEATURES_DIR has audio_*.pt files, "
                    f"2) feature_lens matches between vLLM and bridge, "
                    f"3) vLLM rollout completed before training step."
                )

        # HF audio encoder path (fallback when feature dump dir is absent)
        dtype = next(self.audio_model.parameters()).dtype
        with torch.no_grad():
            outputs = self.audio_model(
                input_features.to(dtype),
                feature_lens=feature_lens,
            )
        return outputs.last_hidden_state  # [num_audio_tokens, hidden]

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        # multimodal kwargs
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        image_input_mask: torch.Tensor = None,
        video_second_per_grid: torch.Tensor = None,
        # audio kwargs
        input_features: torch.Tensor = None,
        feature_attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of the Qwen3-Omni Thinker model.

        Args:
            input_ids: [batch, seq] or THD [1, T] text token ids.
            position_ids: optional, otherwise computed from input_ids.
            attention_mask: text attention mask.
            pixel_values: image pixel values (flat, [N_pix, C*P*P]).
            image_grid_thw: [num_images, 3] (T, H, W) per image.
            pixel_values_videos: video pixel values.
            video_grid_thw: [num_videos, 3] (T, H, W) per video.
            image_input_mask: optional precomputed image mask.
            video_second_per_grid: seconds per video grid (for MRoPE t_index).
            input_features: audio mel features [batch, channels, time].
            feature_attention_mask: audio attention mask [batch, time].
        """
        assert inference_params is None, "Inference not supported"

        # Extract cu_seqlens and CP info early
        cu_seqlens = None
        if packed_seq_params is not None:
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
        cp_size = parallel_state.get_context_parallel_world_size()

        # Audio feature lengths (computed once, used by both audio encoder and MRoPE)
        audio_feature_lengths = None
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)

        # Vision bookkeeping
        video_start_index = 0
        vision_grid_thw = None
        vision_data = None
        image_mask = None
        video_mask = None
        deepstack_feature_lists = None

        combined_embeddings = None
        visual_pos_masks = None

        if self.pre_process:
            # =========================
            # Vision (image / video)
            # =========================
            if image_grid_thw is not None or video_grid_thw is not None:
                if image_grid_thw is not None:
                    image_mask = image_input_mask
                    if image_mask is None:
                        image_mask = (input_ids == self.image_token_id).contiguous()
                    vision_grid_thw = image_grid_thw
                    vision_data = pixel_values
                    video_start_index = image_mask.sum().item()
                else:
                    video_start_index = 0

                if video_grid_thw is not None:
                    video_mask = (input_ids == self.video_token_id).contiguous()
                    if vision_grid_thw is not None:
                        vision_grid_thw = torch.cat([vision_grid_thw, video_grid_thw], dim=0)
                        vision_data = torch.cat([vision_data, pixel_values_videos], dim=0)
                    else:
                        vision_grid_thw = video_grid_thw
                        vision_data = pixel_values_videos

            vision_embeds = None
            if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                vision_embeds, deepstack_feature_lists = self._get_vision_features(vision_data, vision_grid_thw)
                vision_embeds = vision_embeds.to(dtype=self.language_model.embedding.word_embeddings.weight.dtype)

            # =========================
            # Text embeddings
            # =========================
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()  # [seq, batch, hidden]

            # =========================
            # Scatter vision embeds
            # =========================
            if vision_embeds is not None:
                if video_start_index == 0:
                    image_embeds = None
                    video_embeds = vision_embeds
                elif video_start_index == vision_embeds.shape[0]:
                    image_embeds = vision_embeds
                    video_embeds = None
                elif 0 < video_start_index < vision_embeds.shape[0]:
                    image_embeds = vision_embeds[:video_start_index]
                    video_embeds = vision_embeds[video_start_index:]
                else:
                    raise ValueError(
                        f"Expect video token start index in range [0, {vision_embeds.shape[0]}], but got "
                        f"{video_start_index}"
                    )

                # [seq, bs, h] -> [bs, seq, h] for masked scatter
                combined_embeddings_bsh = combined_embeddings.transpose(0, 1).contiguous()
                if image_embeds is not None:
                    combined_embeddings_bsh[image_mask] = image_embeds
                if video_embeds is not None:
                    combined_embeddings_bsh[video_mask] = video_embeds
                combined_embeddings = combined_embeddings_bsh.transpose(0, 1).contiguous()

                if image_embeds is not None and video_embeds is not None:
                    visual_pos_masks = image_mask | video_mask
                elif image_embeds is not None:
                    visual_pos_masks = image_mask
                elif video_embeds is not None:
                    visual_pos_masks = video_mask

            # =========================
            # Audio
            # =========================
            if input_features is not None:
                audio_mask = (input_ids == self.audio_token_id).contiguous()
                # Flatten input_features using feature_attention_mask
                if feature_attention_mask is not None:
                    input_features_flat = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
                else:
                    input_features_flat = input_features

                feature_lens = (
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )

                audio_embeds = self._get_audio_features(input_features_flat, feature_lens)
                audio_embeds = audio_embeds.to(combined_embeddings.dtype)

                combined_embeddings_bsh = combined_embeddings.transpose(0, 1).contiguous()
                combined_embeddings_bsh[audio_mask] = audio_embeds
                combined_embeddings = combined_embeddings_bsh.transpose(0, 1).contiguous()

            # Scatter to sequence-parallel region if needed
            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()

        # =========================
        # Compute M-RoPE position IDs
        # =========================
        # position_ids must be available on ALL PP stages for rotary embeddings.
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()

        if position_ids is None:
            if self.pre_process:
                # Reconstruct full input_ids if CP > 1
                if cu_seqlens is not None:
                    if cp_size > 1:
                        full_input_ids = _gather_input_ids_from_cp(input_ids, cu_seqlens)
                    else:
                        full_input_ids = input_ids
                    input_ids_batch_seq = _thd_to_batch_seq(full_input_ids, cu_seqlens)
                else:
                    input_ids_batch_seq = input_ids

                # If no multimodal inputs at all, fall back to pure-text positions
                has_multimodal = (
                    (image_grid_thw is not None and image_grid_thw.numel() > 0)
                    or (video_grid_thw is not None and video_grid_thw.numel() > 0)
                    or audio_feature_lengths is not None
                )

                if has_multimodal:
                    pos_batch_seq, _ = _get_rope_index(
                        spatial_merge_size=self.spatial_merge_size,
                        image_token_id=self.image_token_id,
                        video_token_id=self.video_token_id,
                        audio_token_id=self.audio_token_id,
                        vision_start_token_id=self.vision_start_token_id,
                        audio_start_token_id=self.audio_start_token_id,
                        input_ids=input_ids_batch_seq,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        audio_seqlens=audio_feature_lengths,
                        attention_mask=None,
                        use_audio_in_video=self.use_audio_in_video,
                        second_per_grids=video_second_per_grid,
                        position_id_per_seconds=self.position_id_per_seconds,
                    )
                else:
                    # Pure text: standard 1D positions replicated across 3 dims
                    bs, seq_len = input_ids_batch_seq.shape
                    pos = torch.arange(seq_len, device=input_ids_batch_seq.device).unsqueeze(0).expand(bs, -1)
                    pos_batch_seq = torch.stack([pos, pos, pos], dim=0)  # [3, bs, seq]

                if cu_seqlens is not None:
                    pos_packed = _batch_seq_to_thd(pos_batch_seq.permute(1, 2, 0), cu_seqlens)
                    position_ids = pos_packed.permute(2, 0, 1).contiguous()  # [3, 1, T_global]
                else:
                    position_ids = pos_batch_seq  # [3, bs, seq]
            else:
                # Non-first PP stage: allocate buffer with correct shape
                if cu_seqlens is not None:
                    T = cu_seqlens[-1].item()
                    position_ids = torch.zeros(3, 1, T, dtype=torch.float, device=torch.cuda.current_device())
                else:
                    raise NotImplementedError(
                        "Non-THD position_ids broadcast not yet supported for non-first PP stages"
                    )

            # Broadcast position_ids from first to all PP stages
            if pp_size > 1:
                src = parallel_state.get_pipeline_model_parallel_first_rank()
                torch.distributed.broadcast(
                    position_ids,
                    src=src,
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )

        # =========================
        # Split deepstack features for SP / CP
        # =========================
        if self.config.sequence_parallel and visual_pos_masks is not None and deepstack_feature_lists is not None:
            if self.pg_collection is not None:
                tp_size = self.pg_collection.tp.size()
                tp_rank = self.pg_collection.tp.rank()
            else:
                tp_size = mpu.get_tensor_model_parallel_world_size()
                tp_rank = mpu.get_tensor_model_parallel_rank()
            visual_pos_masks, deepstack_feature_lists = split_deepstack_embs(
                visual_pos_masks,
                deepstack_feature_lists,
                tp_size=tp_size,
                tp_rank=tp_rank,
                cp_size=1,
                cp_rank=0,
                sequence_parallel=True,
            )

        # =========================
        # Set is_thd_format to skip CP slicing in Qwen3VLMultimodalRotaryEmbedding
        # =========================
        # Standard Qwen3-VL model sets is_thd_format=True dynamically (model.py:805,822)
        # when using packed sequences with CP. Bridge must do the same, otherwise
        # Qwen3VLMultimodalRotaryEmbedding.forward slices emb along CP (because
        # packed_seq kwarg is swallowed by **kwargs and is_thd_format stays False),
        # producing freqs with T_global/cp_size entries. Then _apply_rotary_pos_emb_thd
        # CASE 2 (_get_thd_freqs_on_this_cp_rank) accesses out-of-bounds indices for
        # long sequences, producing a shorter freqs_packed that mismatches t.
        # Fix: set is_thd_format=True for packed (THD) sequences so CP slicing is
        # skipped here; _apply_rotary_pos_emb_thd handles CP per-sequence internally.
        if hasattr(self.language_model, "rotary_pos_emb") and hasattr(
            self.language_model.rotary_pos_emb, "is_thd_format"
        ):
            self.language_model.rotary_pos_emb.is_thd_format = cu_seqlens is not None

        # =========================
        # Language model forward
        # =========================
        # NOTE: visual_pos_masks and deepstack_visual_embeds are Qwen3-Omni-specific
        # DeepStack parameters. Standard Megatron GPTModel does not accept them; they
        # require custom decoder layers. Only pass when not None so text-only training
        # works with the standard GPTModel. Visual inputs need custom decoder support.
        language_model_kwargs = dict(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
        )
        if visual_pos_masks is not None:
            language_model_kwargs["visual_pos_masks"] = visual_pos_masks
        if deepstack_feature_lists is not None:
            language_model_kwargs["deepstack_visual_embeds"] = deepstack_feature_lists
        if extra_block_kwargs:
            language_model_kwargs.update(extra_block_kwargs)

        output = self.language_model(**language_model_kwargs)

        return output


# ---------------------------------------------------------------------------
# Model Provider (dataclass that doubles as TransformerConfig)
# ---------------------------------------------------------------------------
@dataclass
class Qwen3OmniMoeVLModelProvider(GPTModelProvider):
    """Provider that creates Qwen3OmniMoeVLModel.

    Inherits from GPTModelProvider to reuse MoE + TransformerConfig infra.
    Defined at module level (not inside a function) so that the class is
    picklable -- megatron-bridge broadcasts config objects across PP ranks
    via ``torch.distributed.broadcast_object_list`` which requires pickling.

    Default values match Qwen3-Omni-30B-A3B-Instruct.
    """

    # Vision/audio config (stored as HF config objects)
    hf_audio_config: object = None
    hf_vision_config: object = None
    hf_text_config: object = None

    # Multimodal token IDs (defaults from Qwen3-Omni-30B-A3B config)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    audio_token_id: int = 151675
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670

    # Audio MRoPE
    position_id_per_seconds: int = 13
    use_audio_in_video: bool = False

    # M-RoPE
    position_embedding_type: str = "mrope"
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    rotary_base: float = 1000000.0
    rotary_percent: float = 1.0
    scatter_embedding_sequence_parallel: bool = False

    # Vision encoder
    spatial_merge_size: int = 2
    patch_size: int = 16
    temporal_patch_size: int = 2
    deepstack_visual_indexes: list[int] = field(default_factory=lambda: [8, 16, 24])

    # Qwen3 MoE attention
    qk_layernorm: bool = True
    attention_softmax_in_fp32: bool = True
    attention_dropout: float = 0.0

    # MoE router
    moe_router_pre_softmax: bool = False
    moe_router_dtype: str = "fp32"
    moe_router_score_function: str = "softmax"
    moe_router_bias_update_rate: float = 0.001
    moe_permute_fusion: bool = True
    moe_token_dispatcher_type: str = "alltoall"

    # MoE layer layout: all layers MoE by default
    mlp_only_layers: list[int] = field(default_factory=list)
    decoder_sparse_step: int = 1

    # RL: freeze encoders, train only language model
    freeze_language_model: bool = False
    freeze_vision_model: bool = True
    freeze_vision_projection: bool = False
    freeze_audio_model: bool = True

    language_max_sequence_length: int = 32768

    # Performance
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    masked_softmax_fusion: bool = False
    deallocate_pipeline_outputs: bool = True
    distribute_saved_activations: bool = False
    cp_comm_type: str = "p2p"

    def finalize(self) -> None:
        if (self.context_parallel_size or 1) > 1:
            self.calculate_per_token_loss = True
        if self.tensor_model_parallel_size > 1:
            self.sequence_parallel = True
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Create a Qwen3OmniMoeVLModel instance."""
        if pre_process is None:
            pre_process = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
        if post_process is None:
            post_process = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)

        language_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=self.num_moe_experts,
            moe_grouped_gemm=True,
            qk_layernorm=self.qk_layernorm,
            fp8=False,
        )

        model = Qwen3OmniMoeVLModel(
            language_transformer_config=self,
            language_transformer_layer_spec=language_transformer_layer_spec,
            hf_audio_config=self.hf_audio_config,
            hf_vision_config=self.hf_vision_config,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=getattr(self, "_pg_collection", None),
        )

        # Apply freeze options
        if (
            self.freeze_language_model
            or self.freeze_vision_model
            or self.freeze_vision_projection
            or self.freeze_audio_model
        ):
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
                freeze_audio_model=self.freeze_audio_model,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Provide just the language MoE model component without vision/audio."""
        return GPTModelProvider.provide(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


# Add freeze method to Qwen3OmniMoeVLModel via monkey-patch (since it's defined above)
def _qwen3_omni_freeze(
    self,
    freeze_language_model: bool,
    freeze_vision_model: bool,
    freeze_vision_projection: bool,
    freeze_audio_model: bool = False,
):
    """Freeze model modules."""
    if freeze_language_model and self.language_model is not None:
        for param in self.language_model.parameters():
            param.requires_grad = False
    if freeze_vision_model and self.vision_model is not None:
        for param in self.vision_model.parameters():
            param.requires_grad = False
    if freeze_audio_model and self.audio_model is not None:
        for param in self.audio_model.parameters():
            param.requires_grad = False


Qwen3OmniMoeVLModel.freeze = _qwen3_omni_freeze


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------
try:
    from transformers import Qwen3OmniMoeForConditionalGeneration as _Qwen3OmniMoeHF
except ImportError:
    _Qwen3OmniMoeHF = "Qwen3OmniMoeForConditionalGeneration"


@MegatronModelBridge.register_bridge(source=_Qwen3OmniMoeHF, target=Qwen3OmniMoeVLModel)
class Qwen3OmniMoeBridge(MegatronModelBridge):
    """Bridge between HuggingFace Qwen3-Omni-MoE and the Megatron VL model.

    Handles conversion between HuggingFace ``Qwen3OmniMoeForConditionalGeneration``
    (thinker_config) and Megatron ``Qwen3OmniMoeVLModel`` formats.

    Weight mappings follow the Qwen3-VL MoE bridge pattern with the
    ``thinker.`` prefix for HF side (since the HF model is a Thinker-Talker
    container and we only load the Thinker part).
    """

    def provider_bridge(self, hf_pretrained):
        """Create a Qwen3OmniMoeVLModelProvider from HF config."""
        hf_config = hf_pretrained.config.thinker_config
        text_config = hf_config.text_config

        model_dtype = self.dtype_from_hf(text_config, default=torch.bfloat16)

        audio_config = deepcopy(hf_config.audio_config)
        audio_config.torch_dtype = model_dtype
        vision_config = deepcopy(hf_config.vision_config)
        vision_config.torch_dtype = model_dtype

        rope_params = getattr(text_config, "rope_parameters", {}) or getattr(text_config, "rope_scaling", {}) or {}
        mrope_section = rope_params.get("mrope_section", [24, 20, 20])
        rotary_base = rope_params.get("rope_theta", getattr(text_config, "rope_theta", 1000000.0))

        head_dim = getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )

        provider = Qwen3OmniMoeVLModelProvider(
            # Language model
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            moe_ffn_hidden_size=text_config.moe_intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=head_dim,
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            normalization="RMSNorm",
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            hidden_dropout=0.0,
            autocast_dtype=model_dtype,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=rotary_base,
            rotary_percent=1.0,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            # MoE
            num_moe_experts=text_config.num_experts,
            moe_router_topk=text_config.num_experts_per_tok,
            moe_grouped_gemm=True,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=getattr(text_config, "router_aux_loss_coef", 1e-3),
            decoder_sparse_step=getattr(text_config, "decoder_sparse_step", 1),
            mlp_only_layers=getattr(text_config, "mlp_only_layers", []),
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            moe_router_pre_softmax=False,
            moe_router_score_function="softmax",
            moe_router_dtype="fp32",
            # Attention
            add_qkv_bias=getattr(text_config, "attention_bias", False),
            qk_layernorm=getattr(text_config, "use_qk_norm", True),
            # M-RoPE
            mrope_section=mrope_section,
            position_embedding_type="mrope",
            scatter_embedding_sequence_parallel=False,
            # Vision
            spatial_merge_size=getattr(vision_config, "spatial_merge_size", 2),
            patch_size=getattr(vision_config, "patch_size", 16),
            temporal_patch_size=getattr(vision_config, "temporal_patch_size", 2),
            deepstack_visual_indexes=getattr(vision_config, "deepstack_visual_indexes", [8, 16, 24]),
            # Audio
            hf_audio_config=audio_config,
            hf_vision_config=vision_config,
            hf_text_config=text_config,
            audio_token_id=getattr(hf_config, "audio_token_id", 151675),
            audio_start_token_id=getattr(hf_config, "audio_start_token_id", 151669),
            audio_end_token_id=getattr(hf_config, "audio_end_token_id", 151670),
            position_id_per_seconds=getattr(hf_config, "position_id_per_seconds", 13),
            # Vision tokens
            image_token_id=getattr(hf_config, "image_token_id", 151655),
            video_token_id=getattr(hf_config, "video_token_id", 151656),
            vision_start_token_id=getattr(hf_config, "vision_start_token_id", 151652),
            vision_end_token_id=getattr(hf_config, "vision_end_token_id", 151653),
            language_max_sequence_length=text_config.max_position_embeddings,
        )

        # Set head_dim after construction (not a constructor param for GPTModelProvider)
        provider.head_dim = head_dim

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Weight mappings from HF Qwen3-Omni-MoE to Megatron format.

        HF side uses ``thinker.`` prefix (we only load the Thinker part).
        Megatron side uses ``language_model.`` / ``vision_model.`` / ``audio_model.`` prefixes.

        Layer 0+ are all MoE (decoder_sparse_step=1, mlp_only_layers=[]).
        """
        param_mappings = {
            # Embeddings and output
            "language_model.embedding.word_embeddings.weight": "thinker.model.embed_tokens.weight",
            "language_model.output_layer.weight": "thinker.lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "thinker.model.norm.weight",
            # Attention: input layernorm (TE format - fused into linear)
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "thinker.model.layers.*.input_layernorm.weight",
            # Separate input layernorm (non-TE/quantization format)
            "language_model.decoder.layers.*.input_layernorm.weight": "thinker.model.layers.*.input_layernorm.weight",
            # Pre-MLP layernorm
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "thinker.model.layers.*.post_attention_layernorm.weight",
            # Dense MLP layer norm (for non-MoE layers, i.e. mlp_only_layers)
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "thinker.model.layers.*.post_attention_layernorm.weight",
            # Attention output projection
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "thinker.model.layers.*.self_attn.o_proj.weight",
            # QK layernorm weights (Qwen3 specific)
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "thinker.model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "thinker.model.layers.*.self_attn.k_norm.weight",
            # MoE router weights
            "language_model.decoder.layers.*.mlp.router.weight": "thinker.model.layers.*.mlp.gate.weight",
            # MoE router expert bias
            "language_model.decoder.layers.*.mlp.router.expert_bias": "thinker.model.layers.*.mlp.gate.e_score_correction_bias",
            # Dense MLP down projection (for non-MoE layers)
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "thinker.model.layers.*.mlp.down_proj.weight",
            # MoE expert down (TEGroupedMLP: weight* suffix → per-expert HF keys)
            "language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*": "thinker.model.layers.*.mlp.experts.*.down_proj.weight",
            # Shared expert down projection (if present)
            "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "thinker.model.layers.*.mlp.shared_experts.down_proj.weight",
            # Shared expert gate weight (Qwen3-Omni-MoE uses shared_expert_gate)
            "language_model.decoder.layers.*.mlp.shared_experts.gate_weight": "thinker.model.layers.*.mlp.shared_expert_gate.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                # Audio model weights - replicated directly (HF encoder)
                ReplicatedMapping(
                    megatron_param="audio_model.**",
                    hf_param="thinker.audio_tower.**",
                ),
                # Vision model weights - replicated directly (HF encoder)
                ReplicatedMapping(
                    megatron_param="vision_model.**",
                    hf_param="thinker.visual.**",
                ),
                # QKV weight: Combine separate Q, K, V matrices
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="thinker.model.layers.*.self_attn.q_proj.weight",
                    k="thinker.model.layers.*.self_attn.k_proj.weight",
                    v="thinker.model.layers.*.self_attn.v_proj.weight",
                ),
                # QKV bias mapping (if attention_bias is True)
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="thinker.model.layers.*.self_attn.q_proj.bias",
                    k="thinker.model.layers.*.self_attn.k_proj.bias",
                    v="thinker.model.layers.*.self_attn.v_proj.bias",
                ),
                # MoE expert gate+up (TEGroupedMLP) — per-expert HF keys (same as glm4v_moe)
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="thinker.model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                # Expert mappings for SequentialMLP (non-grouped, e.g. for quantization)
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="thinker.model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="thinker.model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # Dense MLP gate+up (for non-MoE layers, i.e. mlp_only_layers)
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="thinker.model.layers.*.mlp.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.up_proj.weight",
                ),
                # Shared expert gate+up (if shared_experts exists)
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="thinker.model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
