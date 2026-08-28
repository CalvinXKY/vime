"""
Colocated vLLM weight sync (trainer side)
=========================================

``UpdateWeightFromTensor`` — Megatron → HF chunks → CUDA IPC handles
→ ``POST /update_weights`` to vLLM's native ``IPCWeightTransferEngine``.

vLLM handles UUID routing + device_index remapping + layerwise reload
internally; no worker extension or monkey-patch is needed.

https://docs.vllm.ai/en/stable/examples/rl/rlhf_ipc/
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray

logger = logging.getLogger(__name__)
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from vime.utils.distributed_utils import get_gloo_group

from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)

_MAX_COLOCATED_UPDATES_INFLIGHT = 4


def _build_packed_ipc_update_info(
    named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> tuple[dict[str, Any], torch.Tensor | None]:
    if not named_tensors:
        return (
            {
                "names": [],
                "dtype_names": [],
                "shapes": [],
                "tensor_sizes": [],
                "ipc_handles": {},
            },
            None,
        )

    from torch.multiprocessing.reductions import reduce_tensor
    from vllm.distributed.weight_transfer.packed_tensor import pack_tensors

    chunk = pack_tensors(
        iter(named_tensors),
        post_iter_func=lambda item: item[1],
        buffer_size_bytes=sum(tensor.numel() * tensor.element_size() for _, tensor in named_tensors),
    )
    assert chunk is not None
    _, ipc_args = reduce_tensor(chunk.packed_tensor)
    gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    return (
        {
            "names": chunk.names,
            "dtype_names": [str(dtype).split(".")[-1] for dtype in chunk.dtypes],
            "shapes": chunk.shapes,
            "tensor_sizes": chunk.tensor_sizes,
            "ipc_handles": {gpu_uuid: ipc_args},
        },
        chunk.packed_tensor,
    )


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: build CUDA IPC handles → all_gather_object(Gloo CPU, over the engine
    slot ranks) → Ray IPC to engine.  Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Compute param buckets.  IPC Gloo groups are created later in
        ``connect_rollout_engines`` once ``engine_gpu_counts`` is known.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._model_update_groups = None
        # vLLM #39212 IPC transfer-engine init runs once per set of colocated engines.
        self._ipc_initialized = False
        # DSpark draft weights cached to CPU before torch_memory_saver.pause(),
        # because draft model params are NOT covered by weights_backuper
        # (only main model params are backed up).  Accessing offloaded draft
        # tensors after pause() causes CUDA error: invalid argument.
        self._dspark_draft_cpu_cache: list[tuple[str, torch.Tensor]] | None = None
        # vLLM IPC handle payloads may use cloudpickle on the Ray/HTTP bridge.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "vime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Create IPC Gloo gather groups (only on first call; partitioning is
        # fixed across reconnects).
        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = colocate_gpu_offsets[i]

        # Map training ranks to colocated engine actors.
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

        # vLLM #39212: one-time IPC transfer-engine init on each colocated engine.
        if dist.get_rank() == 0 and self.rollout_engines and not self._ipc_initialized:
            ray.get(
                [
                    engine.init_weight_transfer_engine.remote({"init_info": {"packed": True}})
                    for engine in self.rollout_engines
                ]
            )
            self._ipc_initialized = True

    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Empty under colocate today;
        kept symmetric with UpdateWeightFromDistributed so the actor can drain unconditionally.
        """
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    # ------------------------------------------------------------------
    # weight update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        # DSpark draft weight sync: use the CPU cache that was populated
        # before torch_memory_saver.pause() (in cache_dspark_draft_weights).
        # The draft model params are NOT covered by weights_backuper, so
        # we must cache them to CPU while the model is still on GPU.
        dspark_draft_cpu_weights = self._dspark_draft_cpu_cache
        if dspark_draft_cpu_weights is not None:
            logger.info(f"[DSpark] Using cached draft weights ({len(dspark_draft_cpu_weights)} tensors)")
        self._dspark_draft_cpu_cache = None  # consume the cache

        # vLLM #39212: enter weight-update mode on each slot leader.
        if self._ipc_engine is not None and rank == self._ipc_gather_src:
            ray.get(self._ipc_engine.start_weight_update.remote())
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()
        self._send_weight_chunks(megatron_local_weights)

        dist.barrier(group=get_gloo_group())
        # After the barrier all engines have returned, so every rank's last-chunk
        # IPC handles are now released by the consumers.  Clean them up.
        torch.cuda.ipc_collect()

        # vLLM #39212: exit weight-update mode.
        if self._ipc_engine is not None and rank == self._ipc_gather_src:
            ray.get(self._ipc_engine.finish_weight_update.remote())
        dist.barrier(group=get_gloo_group())

        if (
            not self.use_distribute
            and self.args.enable_mtp_training
            and (self.args.vllm_speculative_config or {}).get("method") == "mtp"
        ):
            if self._ipc_engine is not None and rank == self._ipc_gather_src:
                ray.get(self._ipc_engine.start_draft_weight_update.remote())
            dist.barrier(group=get_gloo_group())

            self._send_weight_chunks(megatron_local_weights)

            dist.barrier(group=get_gloo_group())
            torch.cuda.ipc_collect()
            if self._ipc_engine is not None and rank == self._ipc_gather_src:
                ray.get(self._ipc_engine.finish_weight_update.remote())
            dist.barrier(group=get_gloo_group())

        # DSpark draft weight sync: send pre-exported CPU weights to vLLM.
        if dspark_draft_cpu_weights is not None:
            self._sync_dspark_draft_weights_direct(dspark_draft_cpu_weights)

        # int4/fp4 post_process
        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_weight_chunks(self, megatron_local_weights) -> None:
        max_inflight = 1 if self.use_distribute else _MAX_COLOCATED_UPDATES_INFLIGHT
        pending = []
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, weight_refs = self._send_hf_params(hf_named_tensors)
            pending.append((refs, weight_refs))
            if len(pending) >= max_inflight:
                self._drain_ipc_updates(pending)
        self._drain_ipc_updates(pending)

    @torch.no_grad()
    def cache_dspark_draft_weights(self) -> None:
        """Export DSpark draft weights to CPU and cache for later use.

        MUST be called BEFORE torch_memory_saver.pause() (i.e., before
        ``actor.sleep()``), while the model is still on GPU.  The cached
        CPU tensors are consumed by ``update_weights()`` after the model
        has been offloaded.

        This mirrors what ``weights_backuper.backup("actor")`` does for
        the main model — but the draft model is NOT covered by the
        weights_backuper, so we handle it separately.
        """
        if not (
            getattr(self.args, "dspark_enabled", False)
            and (self.args.vllm_speculative_config or {}).get("method") == "dspark"
        ):
            return

        # _ipc_engine and use_distribute are set in connect_rollout_engines,
        # which runs in update_weights() BEFORE the first train_actor() call
        # if engines already exist.  On the first training step, _ipc_engine
        # may be None — skip caching; vLLM already has the initial draft
        # checkpoint weights and no sync is needed until after step 1.
        if self._ipc_engine is None or getattr(self, "use_distribute", False):
            self._dspark_draft_cpu_cache = None
            return

        from vime.backends.megatron_utils.dspark.export import (
            export_dspark_weights_to_hf,
            get_dspark_model_from_policy_chunk,
            get_policy_chunk_with_draft,
        )

        rank = dist.get_rank()
        if rank != self._ipc_gather_src:
            self._dspark_draft_cpu_cache = None
            return

        draft_model = get_dspark_model_from_policy_chunk(self.model)
        if draft_model is None:
            logger.warning("[DSpark] draft_model is None, skipping draft weight sync")
            self._dspark_draft_cpu_cache = None
            return

        policy_chunk = get_policy_chunk_with_draft(self.model)
        # When using pre-trained DSpark checkpoint, the draft model has its own
        # untied embed_tokens/lm_head (loaded from pre-trained, very different
        # from policy's tied weights). Export draft's own copies, not policy's.
        _use_policy_embed = not getattr(self.args, "dspark_pretrained_model", None)
        draft_named_tensors = export_dspark_weights_to_hf(
            draft_model=draft_model,
            policy_model=policy_chunk if _use_policy_embed else None,
        )

        if not draft_named_tensors:
            logger.warning("[DSpark] draft_named_tensors is empty, skipping draft weight sync")
            self._dspark_draft_cpu_cache = None
            return

        total_bytes = sum(t.numel() * t.element_size() for _, t in draft_named_tensors)
        logger.info(
            f"[DSpark] Caching draft weights: {len(draft_named_tensors)} tensors, "
            f"{total_bytes / 1024 / 1024:.1f} MB → CPU"
        )

        torch.cuda.synchronize()

        cpu_weights = []
        for name, t in draft_named_tensors:
            cpu_t = t.detach().to("cpu", copy=True)
            cpu_weights.append((name, cpu_t))
        logger.info(f"[DSpark] Draft weight cache ready ({len(cpu_weights)} tensors)")
        self._dspark_draft_cpu_cache = cpu_weights

    @torch.no_grad()
    def _sync_dspark_draft_weights_direct(self, cpu_weights: list[tuple[str, torch.Tensor]]) -> None:
        """Send pre-exported CPU draft weights to vLLM via file-based transfer.

        Bypasses the IPC weight transfer engine + start_draft_weight_update
        path (which hangs). Instead, saves draft weights to a temp file
        and calls ``load_draft_weights_from_file`` on the gpu_worker via
        the HTTP ``/collective_rpc`` endpoint.
        """
        import tempfile

        rank = dist.get_rank()
        if self._ipc_engine is None or rank != self._ipc_gather_src:
            dist.barrier(group=get_gloo_group())
            return

        total_bytes = sum(t.numel() * t.element_size() for _, t in cpu_weights)
        logger.info(f"[DSpark] Direct sync: {len(cpu_weights)} tensors, " f"{total_bytes / 1024 / 1024:.1f} MB")

        # Save weights to a temp file (shared filesystem between trainer and vLLM).
        weight_dict = {name: t for name, t in cpu_weights}
        tmp_path = tempfile.mktemp(suffix=".pt", prefix="dspark_draft_")
        torch.save(weight_dict, tmp_path)
        logger.info(f"[DSpark] Saved draft weights to {tmp_path}")

        # Call load_draft_weights_from_file on the gpu_worker via HTTP /collective_rpc.
        ray.get(
            self._ipc_engine.load_draft_weights_from_file.remote(
                file_path=tmp_path,
            )
        )
        logger.info("[DSpark] Draft weight direct sync completed")

        # Cleanup temp file.
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        dist.barrier(group=get_gloo_group())

    @torch.no_grad()
    def _send_dspark_draft_weights(self) -> None:
        """Export DSpark draft weights and send to vLLM via IPC.

        Sends only the DSpark draft model's parameters (no policy weights).
        The vLLM engine has already switched to draft model target via
        ``start_draft_weight_update``.
        """
        from vime.backends.megatron_utils.dspark.export import (
            export_dspark_weights_to_hf,
            get_dspark_model_from_policy_chunk,
            get_policy_chunk_with_draft,
        )

        draft_model = get_dspark_model_from_policy_chunk(self.model)
        if draft_model is None:
            logger.warning("[DSpark] draft_model is None, skipping draft weight sync")
            return

        policy_chunk = get_policy_chunk_with_draft(self.model)
        # When using pre-trained DSpark checkpoint, the draft model has its own
        # untied embed_tokens/lm_head (loaded from pre-trained, very different
        # from policy's tied weights). Export draft's own copies, not policy's.
        _use_policy_embed = not getattr(self.args, "dspark_pretrained_model", None)
        draft_named_tensors = export_dspark_weights_to_hf(
            draft_model=draft_model,
            policy_model=policy_chunk if _use_policy_embed else None,
        )

        if not draft_named_tensors:
            logger.warning("[DSpark] draft_named_tensors is empty, skipping draft weight sync")
            return

        # Debug: check tensor devices and dtypes
        total_bytes = sum(t.numel() * t.element_size() for _, t in draft_named_tensors)
        for i, (name, tensor) in enumerate(draft_named_tensors[:5]):
            logger.info(
                f"[DSpark] tensor[{i}] name={name}, device={tensor.device}, dtype={tensor.dtype}, shape={tensor.shape}"
            )
        logger.info(f"[DSpark] ... ({len(draft_named_tensors)} total, {total_bytes / 1024 / 1024:.1f} MB)")

        logger.info(
            f"[DSpark] Sending {len(draft_named_tensors)} draft weight tensors to vLLM, "
            f"ipc_gather_group={self._ipc_gather_group is not None}, "
            f"ipc_engine={self._ipc_engine is not None}"
        )

        # Send in size-bounded chunks to avoid oversized IPC packed tensors.
        # A single 1.8 GB packed IPC buffer can hang packed_ipc_consumer during
        # rebuild_cuda_tensor + clone.  512 MB per chunk keeps each transfer
        # well within safe limits.
        # Use max_inflight=1 for draft weights: concurrent IPC transfers can
        # conflict with torch_memory_saver's emptyCache, causing
        # cudaErrorIllegalAddress during release_block.
        _MAX_CHUNK_BYTES = 512 * 1024 * 1024  # 512 MB
        max_inflight = 1
        pending = []

        chunk: list[tuple[str, torch.Tensor]] = []
        chunk_bytes = 0
        for name, tensor in draft_named_tensors:
            tensor_bytes = tensor.numel() * tensor.element_size()
            if chunk and chunk_bytes + tensor_bytes > _MAX_CHUNK_BYTES:
                refs, weight_refs = self._send_hf_params(chunk)
                logger.info(
                    f"[DSpark] chunk sent: {len(chunk)} tensors, {chunk_bytes / 1024 / 1024:.1f} MB, {len(refs)} refs"
                )
                pending.append((refs, weight_refs))
                if len(pending) >= max_inflight:
                    self._drain_ipc_updates(pending)
                chunk = []
                chunk_bytes = 0
            chunk.append((name, tensor))
            chunk_bytes += tensor_bytes

        if chunk:
            refs, weight_refs = self._send_hf_params(chunk)
            logger.info(
                f"[DSpark] final chunk sent: {len(chunk)} tensors, {chunk_bytes / 1024 / 1024:.1f} MB, {len(refs)} refs"
            )
            pending.append((refs, weight_refs))

        self._drain_ipc_updates(pending)
        logger.info("[DSpark] Draft weight sync completed")

    def _drain_ipc_updates(self, pending) -> None:
        if not pending:
            return
        ray.get([ref for refs, _ in pending for ref in refs])
        if self._ipc_gather_group is not None:
            dist.barrier(group=self._ipc_gather_group)
        # Explicitly drop packed_tensor references and force GC before
        # ipc_collect.  torch_memory_saver's emptyCache (called later by
        # the actor) can trigger cudaErrorIllegalAddress if IPC handles
        # are still open when the caching allocator tries to free blocks.
        import gc

        pending.clear()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    local_info, weight_ref = _build_packed_ipc_update_info(hf_named_tensors)

    slot_size = dist.get_world_size(ipc_gather_group)
    if slot_size <= 1:
        if not local_info["names"]:
            return [], weight_ref
        ref = ipc_engine.update_weights.remote(local_info)
        return [ref], weight_ref

    gathered_infos = [None] * slot_size if dist.get_rank() == ipc_gather_src else None
    dist.gather_object(local_info, object_gather_list=gathered_infos, dst=ipc_gather_src, group=ipc_gather_group)

    refs = []
    if dist.get_rank() == ipc_gather_src:
        if any(info is None for info in gathered_infos):
            raise RuntimeError(f"Missing IPC payloads in slot {ipc_gather_src}; got {gathered_infos!r}")
        rank_local_infos = [info if info["names"] else None for info in gathered_infos]
        if any(info is not None for info in rank_local_infos):
            refs.append(ipc_engine.update_weights.remote(rank_local_infos))

    return refs, weight_ref
