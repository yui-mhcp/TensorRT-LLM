# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-disk cache of the KV cache capacity estimation.

The KV cache capacity estimation builds a temporary executor, warms it up and
runs a ``max_num_tokens`` dummy forward to measure the activation peak, then
tears everything down and builds the final executor. For a given model,
configuration and GPU memory state its outcome — the KV cache byte budget — is
the same at every start. With ``TLLM_KV_CACHE_CAPACITY_CACHE_PATH`` set (or
``TRTLLM_CACHE_DIR``), the budget of an estimation is stored under a key of
the software versions, the GPU, the model and the LLM arguments, together with
the free GPU memory it was estimated from. A later start with the same key and
nearly the same free memory reuses the budget and skips the estimation; the
budget is reduced by any shortfall of free memory, never increased.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import filelock
import torch

from tensorrt_llm.logger import logger

__all__ = ["KvCacheCapacityCache"]

CACHE_PATH_ENV = "TLLM_KV_CACHE_CAPACITY_CACHE_PATH"
_FORMAT_VERSION = 1
# Largest free-memory difference with the estimating run that still reuses its budget.
_FREE_BYTES_TOLERANCE = 128 << 20
_MAX_ENTRIES_PER_KEY = 8


class KvCacheCapacityCache:
    """The cached KV cache budgets of one configuration.

    Args:
        path: The JSON file holding the cache, shared by every configuration.
        key_fields: The JSON-serializable description of the configuration.
    """

    def __init__(self, path: str, key_fields: dict[str, Any]):
        self.path = Path(path).expanduser()
        encoded = json.dumps(key_fields, sort_keys=True, default=str)
        self.key = hashlib.sha256(encoded.encode()).hexdigest()
        self.free_bytes = torch.cuda.mem_get_info()[0]

    @classmethod
    def from_env(cls, key_fields: dict[str, Any]) -> Optional["KvCacheCapacityCache"]:
        """The cache of ``key_fields`` when ``TLLM_KV_CACHE_CAPACITY_CACHE_PATH`` is set."""
        path = os.environ.get(CACHE_PATH_ENV)
        return cls(path, key_fields) if path else None

    def _lock(self) -> filelock.FileLock:
        return filelock.FileLock(f"{self.path}.lock", thread_local=False)

    def _read(self) -> dict[str, Any]:
        try:
            with open(self.path) as f:
                contents = json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            logger.warning(f"Ignoring the unreadable KV cache capacity cache {self.path}: {e}")
            return {}
        if not isinstance(contents, dict) or contents.get("version") != _FORMAT_VERSION:
            return {}
        return contents.get("entries", {})

    def lookup(self) -> Optional[int]:
        """The cached KV cache budget in bytes for the current free memory, if any."""
        with self._lock():
            entries = self._read().get(self.key, [])
        matches = [
            entry
            for entry in entries
            if abs(entry["free_bytes"] - self.free_bytes) <= _FREE_BYTES_TOLERANCE
        ]
        if not matches:
            return None
        entry = min(matches, key=lambda e: abs(e["free_bytes"] - self.free_bytes))
        shortfall = max(0, entry["free_bytes"] - self.free_bytes)
        budget = entry["kv_cache_bytes"] - shortfall
        if budget <= 0:
            return None
        logger.info(
            f"KV cache capacity estimation skipped: reusing the budget of {budget / (1 << 30):.2f} "
            f"GiB cached in {self.path} (estimated with "
            f"{entry['free_bytes'] / (1 << 30):.2f} GiB free, now "
            f"{self.free_bytes / (1 << 30):.2f} GiB)."
        )
        return budget

    def store(self, kv_cache_bytes: int) -> None:
        """Record the budget estimated from the free memory measured at construction."""
        entry = {"free_bytes": self.free_bytes, "kv_cache_bytes": int(kv_cache_bytes)}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock():
                entries = self._read()
                kept = [
                    e
                    for e in entries.get(self.key, [])
                    if abs(e["free_bytes"] - self.free_bytes) > _FREE_BYTES_TOLERANCE
                ]
                entries[self.key] = (kept + [entry])[-_MAX_ENTRIES_PER_KEY:]
                tmp_path = self.path.with_suffix(f".tmp.{os.getpid()}")
                with open(tmp_path, "w") as f:
                    json.dump({"version": _FORMAT_VERSION, "entries": entries}, f, indent=1)
                os.replace(tmp_path, self.path)
        except OSError as e:
            logger.warning(f"Could not write the KV cache capacity cache {self.path}: {e}")
            return
        logger.info(f"KV cache capacity estimation cached in {self.path}.")
