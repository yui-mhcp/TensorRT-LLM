# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

from tensorrt_llm._torch.pyexecutor.weight_snapshot import WeightSnapshot


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 3), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.register_buffer("scale", torch.ones(1))
        self.alpha = 1.0
        self.config = {"a": 1}


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = _Inner()
        self.head = nn.Linear(3, 2, bias=False)


def _load_weights(model: _Model) -> None:
    """A weight loading touching every kind of state a snapshot restores."""
    inner = model.inner
    inner.weight.data.copy_(torch.arange(12, dtype=torch.float32).view(4, 3))
    inner.bias = nn.Parameter(torch.arange(8, dtype=torch.float16), requires_grad=False)
    inner._buffers.pop("scale")
    inner.alpha = 2.5
    inner.config["b"] = 2
    inner.flags = (1, "x", torch.bfloat16)
    inner.first_rows = inner.weight[1:3]
    inner.extra = torch.full((2,), 7.0)
    model.head.weight.data.fill_(3.0)


@pytest.fixture
def snapshot(tmp_path):
    return WeightSnapshot(str(tmp_path), {"model": "test"})


def test_restore_reproduces_the_loaded_state(snapshot):
    loaded = _Model()
    snapshot.begin_capture(loaded)
    _load_weights(loaded)
    snapshot.end_capture(loaded)
    assert snapshot.exists

    restored = _Model()
    head_weight = restored.head.weight
    assert snapshot.restore(restored)

    inner = restored.inner
    assert torch.equal(inner.weight, loaded.inner.weight)
    assert inner.bias.dtype == torch.float16 and torch.equal(inner.bias, loaded.inner.bias)
    assert isinstance(inner.bias, nn.Parameter)
    assert "scale" not in inner._buffers
    assert inner.alpha == 2.5
    assert inner.config == {"a": 1, "b": 2}
    assert inner.flags == (1, "x", torch.bfloat16)
    assert torch.equal(inner.extra, loaded.inner.extra)
    # A view is restored as a view of the restored parameter.
    assert torch.equal(inner.first_rows, loaded.inner.first_rows)
    storage = inner.first_rows.untyped_storage()
    assert storage.data_ptr() == inner.weight.untyped_storage().data_ptr()
    # An unreplaced parameter keeps its object, filled in place.
    assert restored.head.weight is head_weight
    assert torch.equal(head_weight, loaded.head.weight)


def test_unsupported_attribute_disables_the_snapshot(snapshot):
    model = _Model()
    snapshot.begin_capture(model)
    _load_weights(model)
    model.inner.opaque = object()
    snapshot.end_capture(model)
    assert not snapshot.exists


def test_key_depends_on_fields(tmp_path):
    a = WeightSnapshot(str(tmp_path), {"model": "a", "files": {1, 2, 3}})
    assert a.key == WeightSnapshot(str(tmp_path), {"model": "a", "files": {1, 2, 3}}).key
    assert a.key != WeightSnapshot(str(tmp_path), {"model": "b"}).key


def test_new_snapshot_removes_stale_ones_of_the_same_label(tmp_path):
    def save(fields, label):
        model = _Model()
        snapshot = WeightSnapshot(str(tmp_path), fields, label=label)
        snapshot.begin_capture(model)
        _load_weights(model)
        snapshot.end_capture(model)
        return snapshot

    old = save({"version": 1}, "model-a")
    other = save({"version": 1, "model": "b"}, "model-b")
    new = save({"version": 2}, "model-a")
    assert new.exists and other.exists and not old.path.exists()
