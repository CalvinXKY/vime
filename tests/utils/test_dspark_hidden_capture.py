import sys
import types
from contextlib import nullcontext

import torch

from vime.backends.megatron_utils.dspark import hidden_capture


def test_forward_gathers_sequence_parallel_hidden_states(monkeypatch):
    target_hidden = torch.randn(2, 1, 6)
    last_hidden = torch.randn(2, 1, 3)
    gathered = []

    class Capture:
        def __init__(self, *_args):
            pass

        def capture_context(self):
            return nullcontext()

        def get_captured_states(self):
            return hidden_capture.CapturedStates(target_hidden, None, last_hidden)

    class DraftModel:
        config = object()

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return "draft-output"

    draft_model = DraftModel()

    class PolicyModel:
        config = types.SimpleNamespace(sequence_parallel=True)

        def __init__(self):
            self.draft_model = draft_model

        def __call__(self, **_kwargs):
            return "policy-output"

    policy_model = PolicyModel()

    def gather(tensor, *, tensor_parallel_output_grad):
        assert tensor_parallel_output_grad is False
        gathered.append(tensor)
        return torch.cat((tensor, tensor), dim=0)

    core = types.ModuleType("megatron.core")
    core.tensor_parallel = types.SimpleNamespace(gather_from_sequence_parallel_region=gather)
    utils = types.ModuleType("megatron.core.utils")
    utils.unwrap_model = lambda model: model
    megatron = types.ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)
    monkeypatch.setattr(hidden_capture, "HiddenStateCapture", Capture)

    output, draft_output, config = hidden_capture.forward_with_dspark(
        policy_model,
        {},
        {"tokens": torch.ones(1, 4, dtype=torch.long), "full_loss_masks": torch.ones(1, 4)},
        (1,),
    )

    assert output == "policy-output"
    assert draft_output == "draft-output"
    assert config is draft_model.config
    assert gathered[0] is target_hidden
    assert gathered[1] is last_hidden
    assert draft_model.kwargs["target_hidden_states"].shape == (1, 4, 6)
    assert draft_model.kwargs["target_last_hidden_states"].shape == (1, 4, 3)
