# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import enum

import pytest
import torch

from tensorrt_llm._torch import cute_dsl_kernel_cache as kernel_cache


class _Color(enum.Enum):
    RED = 1


class _Kernel:
    pass


class _FakeSpec:
    """A ``JitSpecCuteDsl`` stand-in recording the kernels persisted per name."""

    disk: dict = {}

    def __init__(self, module_name, kernel_name, compile_fn):
        self.name = (module_name, kernel_name)
        self.compile_fn = compile_fn

    @property
    def is_compiled(self):
        return self.name in self.disk

    def try_load(self):
        return self.disk.get(self.name)

    def compile_and_persist(self):
        self.disk[self.name] = ("loaded", self.compile_fn())


@pytest.fixture
def fake_disk(monkeypatch):
    _FakeSpec.disk = {}
    monkeypatch.setattr(kernel_cache, "_make_spec", _FakeSpec)
    monkeypatch.setattr(kernel_cache, "_is_tvm_ffi_function", lambda compiled: True)
    return _FakeSpec.disk


def test_digest_is_stable_and_discriminating():
    key = ((_Kernel,), ("dtype", torch.bfloat16), ("stage", 2), (_Color.RED, None, 1.5, "s"))
    assert kernel_cache._digest(key) == kernel_cache._digest(key)
    assert kernel_cache._digest(key) != kernel_cache._digest(key[:-1])


def test_digest_rejects_unstable_values():
    assert kernel_cache._digest((object(),)) is None


def test_persistent_compile_cache(fake_disk):
    key = (((_Kernel, object), ("stage", 2)), ((str, "sm_120a"),))
    cache = kernel_cache._PersistentCompileCache("module", {})
    assert cache.get(key) is None
    cache[key] = "kernel"
    assert cache.get(key) == "kernel"

    # A new process: the kernel is reloaded from disk.
    assert kernel_cache._PersistentCompileCache("module", {}).get(key) == ("loaded", "kernel")


def test_persistent_compile_cache_skips_unstable_keys(fake_disk):
    key = (((_Kernel,), ("stream", object())), ())
    cache = kernel_cache._PersistentCompileCache("module", {})
    cache[key] = "kernel"
    assert cache.get(key) == "kernel"
    assert not fake_disk


def test_persistent_getter(fake_disk):
    calls = []

    def _get_compiled_norm_kernel(dtype, hidden, *, contiguous=True):
        calls.append((dtype, hidden, contiguous))
        return f"kernel-{dtype}-{hidden}"

    getter = kernel_cache._persistent_getter("module", _get_compiled_norm_kernel)
    assert getter("bf16", 5120, contiguous=False) == "kernel-bf16-5120"
    assert getter("bf16", 5120, contiguous=False) == "kernel-bf16-5120"
    assert len(calls) == 1

    # A new process: the kernel is reloaded from disk, not compiled.
    getter = kernel_cache._persistent_getter("module", _get_compiled_norm_kernel)
    assert getter("bf16", 5120, contiguous=False) == ("loaded", "kernel-bf16-5120")
    assert len(calls) == 1
