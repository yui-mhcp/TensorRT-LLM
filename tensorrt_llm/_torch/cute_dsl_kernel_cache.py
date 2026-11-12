# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent on-disk cache for the FlashInfer CuTe DSL kernels compiled at runtime.

Some FlashInfer CuTe DSL kernels are only cached in memory: the GDN delta-rule
prefill kernels (``gdn_kernels.delta_rule_dsl.custom_compile_cache``) and the
CuTe norm kernels (the ``functools.cache``'d ``_get_compiled_*`` getters of
``flashinfer.norm.kernels``). ``cute.compile`` never caches on disk, so every
process recompiles them during warmup: about 10 s on a Qwen3.5 / Qwen3.8
hybrid model on SM120.

:func:`install` routes both caches through FlashInfer's own persistent
CuTe DSL cache (:class:`flashinfer.jit.cute_dsl_core.JitSpecCuteDsl`): a
compiled kernel is exported to an object file under the FlashInfer JIT
directory, and later processes reload it with ``cute.runtime.load_module``
instead of compiling it. The on-disk key is the in-memory one (the kernel's
compile key and compile options, or the getter's arguments), plus the target
architecture, the CuTe DSL version and the hash of the kernel sources. Keys
that cannot be encoded stably, and kernels that are not TVM-FFI functions,
are left to the in-memory cache.

Set ``TRTLLM_CUTE_DSL_PERSISTENT_CACHE=0`` to disable it, or
``FLASHINFER_CUTE_DSL_DISABLE_CACHE=1`` to disable every FlashInfer CuTe DSL
disk cache.
"""

import enum
import functools
import hashlib
import importlib
import os
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from tensorrt_llm.logger import logger

__all__ = ["install"]

_DISABLE_ENV = "TRTLLM_CUTE_DSL_PERSISTENT_CACHE"
_GDN_MODULE = "flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache"
_NORM_MODULES = (
    "flashinfer.norm.kernels.rmsnorm",
    "flashinfer.norm.kernels.fused_add_rmsnorm",
    "flashinfer.norm.kernels.layernorm",
)
_GETTER_PREFIX = "_get_compiled_"

_installed = False


class _UnstableKey(TypeError):
    """A cache key holding a value without a process-independent encoding."""


def _stable_value(value: object) -> object:
    """A process-independent encoding of ``value``, raising ``_UnstableKey`` otherwise."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, enum.Enum):
        return (type(value).__module__, type(value).__qualname__, value.name)
    if isinstance(value, type):
        return (value.__module__, value.__qualname__)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (tuple, list)):
        return tuple(_stable_value(item) for item in value)
    raise _UnstableKey(type(value).__qualname__)


def _digest(value: object) -> Optional[str]:
    try:
        encoded = repr(_stable_value(value))
    except _UnstableKey:
        return None
    return hashlib.sha256(encoded.encode()).hexdigest()[:32]


def _is_tvm_ffi_function(compiled: Any) -> bool:
    """Whether ``compiled`` can be exported and reloaded through ``load_module``."""
    from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunctionBase

    return isinstance(compiled, TVMFFIJitCompiledFunctionBase)


@functools.cache
def _source_files(module_name: str) -> tuple[str, ...]:
    """The sources keying a module's kernels: its package's files and this file."""
    package = Path(importlib.import_module(module_name).__file__).resolve().parent
    return tuple(sorted(str(path) for path in package.glob("*.py"))) + (
        str(Path(__file__).resolve()),
    )


def _make_spec(module_name: str, kernel_name: str, compile_fn: Callable[[], Any]):
    """The disk cache entry of a kernel of the FlashInfer module ``module_name``."""
    from flashinfer.jit.cute_dsl_core import JitSpecCuteDsl, _hash_source_files

    # Prefixes the exported C symbol: no dots.
    spec_module = "trtllm_" + module_name.removeprefix("flashinfer.").replace(".", "_")
    spec = JitSpecCuteDsl(
        spec_module, kernel_name, compile_fn, _hash_source_files(_source_files(module_name))
    )
    # The version lists every distribution found on sys.path: a process with a duplicated
    # sys.path entry (e.g. an MPI worker) lists each one twice and would invalidate the
    # kernels of the other processes.
    version = spec.expected_meta["cute_dsl_version"]
    spec.expected_meta["cute_dsl_version"] = ";".join(sorted(set(version.split(";"))))
    return spec


class _PersistentCompileCache(dict):
    """The delta-rule ``_in_mem_compile_cache``, backed by the CuTe DSL disk cache.

    ``custom_compile_cache`` only reads the dict with ``get`` and writes it with
    item assignment: a miss in memory is looked up on disk, and a kernel stored
    in memory is exported to disk.
    """

    def __init__(self, module_name: str, initial: dict):
        super().__init__(initial)
        self._module_name = module_name

    def _kernel_name(self, key: tuple) -> Optional[str]:
        digest = _digest(key)
        if digest is None:
            return None
        # key = ((type(kernel).__mro__, *attributes), options)
        return f"{key[0][0][0].__name__}_{digest}"

    def get(self, key, default=None):
        compiled = super().get(key)
        if compiled is not None:
            return compiled
        kernel_name = self._kernel_name(key)
        if kernel_name is None:
            return default
        compiled = _make_spec(self._module_name, kernel_name, lambda: None).try_load()
        if compiled is None:
            return default
        super().__setitem__(key, compiled)
        return compiled

    def __setitem__(self, key, compiled):
        super().__setitem__(key, compiled)
        kernel_name = self._kernel_name(key)
        if kernel_name is None or not _is_tvm_ffi_function(compiled):
            return
        spec = _make_spec(self._module_name, kernel_name, lambda: compiled)
        if not spec.is_compiled:
            # Persistence is best effort: on failure the kernel is recompiled next run.
            spec.compile_and_persist()


def _persistent_getter(module_name: str, getter: Callable[..., Any]) -> Callable[..., Any]:
    """A ``functools.cache``'d kernel getter whose kernels persist on disk."""

    @functools.cache
    @functools.wraps(getter)
    def persistent(*args, **kwargs):
        digest = _digest((args, sorted(kwargs.items())))
        if digest is None:
            return getter(*args, **kwargs)
        kernel_name = f"{getter.__name__[len(_GETTER_PREFIX):]}_{digest}"
        spec = _make_spec(module_name, kernel_name, lambda: getter(*args, **kwargs))
        compiled = spec.try_load()
        if compiled is not None:
            return compiled
        compiled = getter(*args, **kwargs)
        if _is_tvm_ffi_function(compiled):
            spec = _make_spec(module_name, kernel_name, lambda: compiled)
            spec.compile_and_persist()
        return compiled

    return persistent


def _install_gdn() -> None:
    module = importlib.import_module(_GDN_MODULE)
    cache = module._in_mem_compile_cache
    if not isinstance(cache, _PersistentCompileCache):
        module._in_mem_compile_cache = _PersistentCompileCache(_GDN_MODULE, cache)


def _install_norms() -> None:
    for module_name in _NORM_MODULES:
        module = importlib.import_module(module_name)
        for name in dir(module):
            getter = getattr(module, name)
            original = getattr(getter, "__wrapped__", None)
            if not name.startswith(_GETTER_PREFIX) or original is None:
                continue
            if getattr(getter, "_trtllm_persistent", False):
                continue
            persistent = _persistent_getter(module_name, original)
            persistent._trtllm_persistent = True
            setattr(module, name, persistent)


def install() -> None:
    """Back the in-memory FlashInfer CuTe DSL kernel caches with the disk cache.

    Idempotent; a no-op when disabled or when a FlashInfer module is missing.
    Must run before the first kernel compile (the caches are replaced, not merged).
    """
    global _installed
    if _installed or os.environ.get(_DISABLE_ENV, "1") == "0":
        return
    from flashinfer.jit.cute_dsl_core import cute_dsl_cache_disabled

    if cute_dsl_cache_disabled():
        return
    for install_fn in (_install_gdn, _install_norms):
        try:
            install_fn()
        except ImportError as e:
            logger.debug(f"CuTe DSL persistent cache: {install_fn.__name__} skipped ({e})")
    _installed = True
