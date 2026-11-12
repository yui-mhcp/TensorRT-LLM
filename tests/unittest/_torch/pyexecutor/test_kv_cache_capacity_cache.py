# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import kv_cache_capacity_cache as capacity_cache
from tensorrt_llm._torch.pyexecutor.kv_cache_capacity_cache import KvCacheCapacityCache

MIB = 1 << 20


@pytest.fixture
def free_bytes(monkeypatch):
    """Set the free GPU memory the cache measures: ``free_bytes(value)``."""
    state = {"free": 10_000 * MIB}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (state["free"], 32_000 * MIB))

    def set_free(value):
        state["free"] = value

    return set_free


def test_from_env_disabled(monkeypatch, free_bytes):
    monkeypatch.delenv(capacity_cache.CACHE_PATH_ENV, raising=False)
    assert KvCacheCapacityCache.from_env({"model": "a"}) is None


def test_store_then_lookup(monkeypatch, tmp_path, free_bytes):
    path = tmp_path / "kv" / "cache.json"
    monkeypatch.setenv(capacity_cache.CACHE_PATH_ENV, str(path))
    KvCacheCapacityCache.from_env({"model": "a"}).store(4_000 * MIB)

    assert KvCacheCapacityCache.from_env({"model": "a"}).lookup() == 4_000 * MIB
    assert KvCacheCapacityCache.from_env({"model": "b"}).lookup() is None


def test_lookup_reduces_budget_by_shortfall(tmp_path, free_bytes):
    path = tmp_path / "cache.json"
    KvCacheCapacityCache(path, {"model": "a"}).store(4_000 * MIB)

    free_bytes(10_000 * MIB - 100 * MIB)
    assert KvCacheCapacityCache(path, {"model": "a"}).lookup() == 3_900 * MIB
    free_bytes(10_000 * MIB + 100 * MIB)
    assert KvCacheCapacityCache(path, {"model": "a"}).lookup() == 4_000 * MIB
    free_bytes(10_000 * MIB - 200 * MIB)
    assert KvCacheCapacityCache(path, {"model": "a"}).lookup() is None


def test_store_replaces_close_entries(tmp_path, free_bytes):
    path = tmp_path / "cache.json"
    KvCacheCapacityCache(path, {"model": "a"}).store(4_000 * MIB)
    KvCacheCapacityCache(path, {"model": "a"}).store(4_100 * MIB)
    free_bytes(5_000 * MIB)
    KvCacheCapacityCache(path, {"model": "a"}).store(1_000 * MIB)

    (entries,) = json.loads(path.read_text())["entries"].values()
    assert sorted(entry["kv_cache_bytes"] for entry in entries) == [1_000 * MIB, 4_100 * MIB]


def test_unreadable_cache_is_ignored(tmp_path, free_bytes):
    path = tmp_path / "cache.json"
    path.write_text("not json")
    assert KvCacheCapacityCache(path, {"model": "a"}).lookup() is None
    KvCacheCapacityCache(path, {"model": "a"}).store(4_000 * MIB)
    assert KvCacheCapacityCache(path, {"model": "a"}).lookup() == 4_000 * MIB
