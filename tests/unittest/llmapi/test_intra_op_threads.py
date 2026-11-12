# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tensorrt_llm import _utils
from tensorrt_llm.llmapi import utils


@pytest.fixture
def threads(monkeypatch):
    """A 24-logical / 16-physical CPU machine whose torch runs ``state["threads"]`` threads."""
    state = {"threads": 1, "history": []}

    def set_num_threads(n):
        state["threads"] = n
        state["history"].append(n)

    monkeypatch.setattr(torch, "get_num_threads", lambda: state["threads"])
    monkeypatch.setattr(torch, "set_num_threads", set_num_threads)
    monkeypatch.setattr(utils.os, "sched_getaffinity", lambda pid: set(range(24)))
    monkeypatch.setattr(utils.psutil, "cpu_count", lambda logical=True: 24 if logical else 16)
    monkeypatch.setattr(_utils, "local_mpi_size", lambda: 1)
    for name in utils._INTRA_OP_THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return state


def test_single_thread_worker_loads_on_physical_cores(threads):
    with utils.loading_intra_op_threads():
        assert threads["threads"] == 16
    assert threads["threads"] == 1


def test_single_thread_is_restored_on_error(threads):
    with pytest.raises(RuntimeError), utils.loading_intra_op_threads():
        raise RuntimeError("load failed")
    assert threads["threads"] == 1


def test_cores_are_shared_between_local_ranks(threads, monkeypatch):
    monkeypatch.setattr(_utils, "local_mpi_size", lambda: 4)
    with utils.loading_intra_op_threads():
        assert threads["threads"] == 4


def test_explicit_thread_count_is_kept(threads, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with utils.loading_intra_op_threads():
        pass
    assert threads["history"] == []


def test_multi_thread_default_is_kept(threads):
    threads["threads"] = 8
    with utils.loading_intra_op_threads():
        pass
    assert threads["history"] == []
