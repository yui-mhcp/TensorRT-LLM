# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-disk snapshot of the model state produced by the checkpoint weight loading.

``model.load_weights`` turns the checkpoint tensors into the module layout:
name mapping, fusion, packing, requantization, row permutations, then the
copies to the GPU. Its result only depends on the checkpoint, the model
configuration and the software versions, yet it is recomputed at every start
(about 10 s for Qwen3.8-27B NVFP4, most of it CPU work and pageable copies).

With ``TLLM_WEIGHT_SNAPSHOT_DIR`` set, the first load saves the state right
after ``model.load_weights`` — every parameter and buffer, plus the module
attributes the loading set or changed — and later loads copy it back into the
freshly constructed model instead of reading the checkpoint. Everything after
``load_weights`` (the post-load hooks) runs as usual on both paths.

A snapshot is only written when every changed attribute can be restored: a
tensor (saved, or rebuilt as a view of a saved tensor) or a plain value
(``None``, bool, int, float, str, ``torch.dtype`` or tuples / lists of them).
It is keyed by the software versions, the checkpoint files (names, sizes and
modification times), the model configuration and the mapping, and stored as a
raw tensor file plus a JSON index.
"""

import enum
import hashlib
import json
import mmap
import os
import shutil
import weakref
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from tensorrt_llm.logger import logger

__all__ = ["WeightSnapshot"]

SNAPSHOT_DIR_ENV = "TLLM_WEIGHT_SNAPSHOT_DIR"
_FORMAT_VERSION = 1
_INDEX_FILE = "index.json"
_DATA_FILE = "tensors.bin"
_ALIGNMENT = 4096
# nn.Module bookkeeping, never part of the loaded state.
_MODULE_INTERNALS = frozenset({"training"})
_PLAIN_TYPES = (type(None), bool, int, float, str)
_DTYPES = {
    str(dtype): dtype
    for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16,
                  torch.float8_e4m3fn, torch.float8_e5m2, torch.int64,
                  torch.int32, torch.int16, torch.int8, torch.uint8,
                  torch.bool)
}
_DTYPES.update({str(dtype): dtype
                for dtype in (getattr(torch, "float8_e8m0fnu", None),
                              getattr(torch, "float4_e2m1fn_x2", None))
                if dtype is not None})


class _Unsupported(TypeError):
    """A loaded attribute without a snapshot encoding."""


def _encode_plain(value: Any) -> Any:
    if isinstance(value, _PLAIN_TYPES):
        return {"plain": value}
    if isinstance(value, torch.dtype) and str(value) in _DTYPES:
        return {"dtype": str(value)}
    if isinstance(value, (tuple, list)):
        return {
            "tuple" if isinstance(value, tuple) else "list":
            [_encode_plain(item) for item in value]
        }
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"dict": {k: _encode_plain(v) for k, v in value.items()}}
    raise _Unsupported(type(value).__qualname__)


def _decode_plain(encoded: dict) -> Any:
    (kind, value), = encoded.items()
    if kind == "plain":
        return value
    if kind == "dtype":
        return _DTYPES[value]
    if kind == "dict":
        return {k: _decode_plain(v) for k, v in value.items()}
    items = [_decode_plain(item) for item in value]
    return tuple(items) if kind == "tuple" else items


class _Opaque:
    """An attribute value without encoding, compared by identity."""

    def __init__(self, value: Any):
        self.value = value

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Opaque) and other.value is self.value


def _attribute_state(model: nn.Module) -> dict[str, dict[str, Any]]:
    """The plain (non-parameter, non-buffer, non-module) attributes of every module.

    Tensors are kept as is, encodable values as their encoding (a container
    mutated in place then compares different), anything else as ``_Opaque``.
    """
    state = {}
    for name, module in model.named_modules(remove_duplicate=False):
        attributes = {}
        for attr, value in vars(module).items():
            if attr.startswith("_") or attr in _MODULE_INTERNALS:
                continue
            if isinstance(value, (torch.Tensor, nn.Module)):
                attributes[attr] = value
                continue
            try:
                attributes[attr] = _encode_plain(value)
            except _Unsupported:
                attributes[attr] = _Opaque(value)
        state[name] = attributes
    return state


def _tensor_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """The parameters and buffers of every module, by qualified name."""
    tensors = {}
    for name, module in model.named_modules(remove_duplicate=False):
        prefix = f"{name}." if name else ""
        for attr, tensor in list(module._parameters.items()) + list(
                module._buffers.items()):
            if tensor is not None:
                tensors[prefix + attr] = tensor
    return tensors


def _storage_key(tensor: torch.Tensor) -> Optional[int]:
    """The identity of a tensor's storage; None for an empty one (never aliased)."""
    storage = tensor.untyped_storage()
    return storage.data_ptr() if storage.nbytes() > 0 else None


def _normalize(value: Any) -> Any:
    """A JSON value of ``value`` independent of hash ordering (sets are sorted)."""
    if isinstance(value, _PLAIN_TYPES):
        return value
    if isinstance(value, enum.Enum):
        return _normalize(value.value)
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_normalize(item) for item in value), key=repr)
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _normalize(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _normalize(to_dict())
    return str(value)


def _source_fingerprint() -> str:
    """The sizes and modification times of the sources the weight loading runs."""
    package = Path(__file__).resolve().parents[2]
    roots = [package / "_torch" / "models", package / "_torch" / "modules",
             package / "quantization"]
    files = [Path(__file__).resolve(), package / "_torch" / "pyexecutor" / "model_loader.py"]
    for root in roots:
        files.extend(root.rglob("*.py"))
    sha = hashlib.sha256()
    for path in sorted(files):
        stat = path.stat()
        sha.update(f"{path.relative_to(package)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return sha.hexdigest()


def _distribution_versions(*names: str) -> dict[str, Optional[str]]:
    from importlib import metadata

    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _layout_fingerprint(model: nn.Module) -> str:
    """The module classes and the parameter / buffer layouts of the constructed model."""
    sha = hashlib.sha256()
    for name, module in model.named_modules(remove_duplicate=False):
        sha.update(f"{name}:{type(module).__module__}.{type(module).__qualname__}\n".encode())
    for name, tensor in _tensor_state(model).items():
        sha.update(f"{name}:{list(tensor.shape)}:{tensor.dtype}\n".encode())
    return sha.hexdigest()


def _checkpoint_fingerprint(checkpoint_dir: str) -> list:
    files = []
    for path in sorted(Path(checkpoint_dir).iterdir()):
        if path.is_file():
            stat = path.stat()
            files.append((path.name, stat.st_size, int(stat.st_mtime)))
    return files


class WeightSnapshot:
    """The snapshot of one model load.

    Args:
        directory: The root of every snapshot (``TLLM_WEIGHT_SNAPSHOT_DIR``).
        key_fields: The JSON-serializable identity of the load.
    """

    def __init__(self,
                 directory: str,
                 key_fields: dict[str, Any],
                 label: Optional[str] = None):
        encoded = json.dumps(key_fields, sort_keys=True, default=str)
        self.key = hashlib.sha256(encoded.encode()).hexdigest()[:32]
        self.root = Path(directory).expanduser()
        self.path = self.root / self.key
        # Snapshots of the same label (checkpoint) under another key are stale.
        self.label = label
        self._before: Optional[dict[str, dict[str, Any]]] = None
        self._before_tensors: Optional[dict[str, weakref.ref]] = None

    @classmethod
    def from_env(cls, *, model: nn.Module, checkpoint_dir: str, config: Any,
                 mapping: Any) -> Optional["WeightSnapshot"]:
        """The snapshot of this load when ``TLLM_WEIGHT_SNAPSHOT_DIR`` is set.

        Args:
            model: The constructed model, before any weight is loaded.
        """
        directory = os.environ.get(SNAPSHOT_DIR_ENV)
        if not directory or not os.path.isdir(checkpoint_dir):
            return None
        import tensorrt_llm

        return cls(
            directory,
            label=str(Path(checkpoint_dir).resolve()),
            key_fields={
                "format": _FORMAT_VERSION,
                "tensorrt_llm": tensorrt_llm.__version__,
                "torch": torch.__version__,
                "distributions": _distribution_versions("flashinfer-python",
                                                        "nvidia-cutlass-dsl"),
                "sources": _source_fingerprint(),
                "device": torch.cuda.get_device_name(),
                "checkpoint": _checkpoint_fingerprint(checkpoint_dir),
                "pretrained_config": _normalize(config.pretrained_config),
                "quant_config": _normalize(config.quant_config),
                "quant_config_dict": _normalize(config.quant_config_dict),
                "mapping": _normalize(mapping),
                "attn_backend": str(getattr(config, "attn_backend", None)),
                "moe_backend": str(getattr(config, "moe_backend", None)),
                "layout": _layout_fingerprint(model),
            })

    @property
    def exists(self) -> bool:
        return (self.path / _INDEX_FILE).is_file()

    # ------------------------------------------------------------- capture

    def begin_capture(self, model: nn.Module) -> None:
        """Record the state of the constructed model before ``load_weights``.

        Only weak references to the tensors are kept: a parameter replaced by
        the loading is freed as usual.
        """
        self._before = {
            module_name: {
                attr: None if isinstance(value, torch.Tensor) else value
                for attr, value in attributes.items()
            }
            for module_name, attributes in _attribute_state(model).items()
        }
        self._before_tensors = {
            name: weakref.ref(tensor)
            for name, tensor in _tensor_state(model).items()
        }

    def end_capture(self, model: nn.Module) -> None:
        """Save the state ``load_weights`` produced; best effort."""
        before, self._before = self._before, None
        before_tensors, self._before_tensors = self._before_tensors, None
        try:
            index, tensors = self._describe(model, before, before_tensors)
        except _Unsupported as e:
            logger.warning(
                f"Weight snapshot disabled for this model: a loaded attribute "
                f"cannot be saved ({e}).")
            return
        try:
            self._write(index, tensors)
        except OSError as e:
            logger.warning(f"Could not write the weight snapshot {self.path}: {e}")
            return
        logger.info(f"Weight snapshot saved to {self.path}.")

    def _describe(self, model, before, before_tensors):
        tensors = _tensor_state(model)
        index: dict[str, Any] = {
            "tensors": {},
            "removed_tensors": sorted(set(before_tensors) - set(tensors)),
            "attributes": {},
        }
        # Saved tensors and the storages they own, to rebuild aliases as views.
        saved: dict[str, torch.Tensor] = {}
        owners: dict[int, str] = {}

        def describe_tensor(name: str, tensor: torch.Tensor) -> dict[str, Any]:
            key = _storage_key(tensor)
            if key in owners:
                return {"view_of": owners[key], **self._view(tensor)}
            if key is not None:
                owners[key] = name
            saved[name] = tensor
            return self._layout(tensor)

        for name, tensor in tensors.items():
            index["tensors"][name] = {
                "parameter": isinstance(tensor, nn.Parameter),
                "requires_grad": tensor.requires_grad,
                "replaced": (name not in before_tensors
                             or before_tensors[name]() is not tensor),
                **describe_tensor(name, tensor),
            }

        after = _attribute_state(model)
        for module_name, attributes in after.items():
            old = before.get(module_name, {})
            changed = {}
            for attr, value in attributes.items():
                if isinstance(value, torch.Tensor):
                    # Saved even when unchanged: it may have been updated in place.
                    changed[attr] = {
                        "tensor": f"{module_name}.{attr}#attr",
                        **describe_tensor(f"{module_name}.{attr}#attr", value)
                    }
                    continue
                if attr in old and (old[attr] is value or old[attr] == value):
                    continue
                if isinstance(value, (nn.Module, _Opaque)):
                    raise _Unsupported(f"{module_name}.{attr}")
                changed[attr] = {"value": value}
            removed = sorted(set(old) - set(attributes))
            if changed or removed:
                index["attributes"][module_name] = {
                    "changed": changed,
                    "removed": removed
                }
        return index, saved

    @staticmethod
    def _layout(tensor: torch.Tensor) -> dict[str, Any]:
        if str(tensor.dtype) not in _DTYPES:
            raise _Unsupported(f"dtype {tensor.dtype}")
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "storage_offset": tensor.storage_offset(),
            "storage_bytes": tensor.untyped_storage().nbytes(),
            "device": str(tensor.device),
        }

    @staticmethod
    def _fits(current: torch.Tensor, entry: dict) -> bool:
        """Whether ``current`` is a whole contiguous storage of the entry's layout."""
        return (str(current.dtype) == entry["dtype"]
                and list(current.shape) == entry["shape"]
                and current.is_contiguous() and current.storage_offset() == 0
                and entry["storage_offset"] == 0
                and current.untyped_storage().nbytes() == entry["storage_bytes"]
                and str(current.device) == entry["device"])

    @staticmethod
    def _view(tensor: torch.Tensor) -> dict[str, Any]:
        if str(tensor.dtype) not in _DTYPES:
            raise _Unsupported(f"dtype {tensor.dtype}")
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "byte_offset": tensor.storage_offset() * tensor.element_size(),
        }

    def _write(self, index: dict, tensors: dict[str, torch.Tensor]) -> None:
        """Stream every saved storage to the data file, one at a time."""
        self.path.mkdir(parents=True, exist_ok=True)
        tmp_data = self.path / f"{_DATA_FILE}.tmp.{os.getpid()}"
        layouts = {
            name: entry
            for name, entry in index["tensors"].items() if "view_of" not in entry
        }
        for module_attrs in index["attributes"].values():
            for entry in module_attrs["changed"].values():
                if "tensor" in entry and "view_of" not in entry:
                    layouts[entry["tensor"]] = entry
        offset = 0
        with open(tmp_data, "wb") as f:
            for name, tensor in tensors.items():
                storage = tensor.untyped_storage()
                host = torch.empty(storage.nbytes(), dtype=torch.uint8)
                host.untyped_storage().copy_(storage)
                padding = -offset % _ALIGNMENT
                f.write(b"\0" * padding)
                offset += padding
                layouts[name]["data_offset"] = offset
                f.write(host.numpy().tobytes())
                offset += storage.nbytes()
        os.replace(tmp_data, self.path / _DATA_FILE)
        tmp_index = self.path / f"{_INDEX_FILE}.tmp.{os.getpid()}"
        with open(tmp_index, "w") as f:
            json.dump({"version": _FORMAT_VERSION, "label": self.label, **index}, f)
        os.replace(tmp_index, self.path / _INDEX_FILE)
        self._remove_stale()

    def _remove_stale(self) -> None:
        """Delete the snapshots of the same label written under another key."""
        if self.label is None:
            return
        for other in self.root.iterdir():
            if other == self.path or not (other / _INDEX_FILE).is_file():
                continue
            try:
                with open(other / _INDEX_FILE) as f:
                    label = json.load(f).get("label")
            except (OSError, ValueError):
                continue
            if label == self.label:
                logger.info(f"Removing the stale weight snapshot {other}.")
                shutil.rmtree(other, ignore_errors=True)

    # ------------------------------------------------------------- restore

    def restore(self, model: nn.Module) -> bool:
        """Load the snapshot into the constructed model; False when unusable."""
        try:
            with open(self.path / _INDEX_FILE) as f:
                index = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning(f"Ignoring the unreadable weight snapshot {self.path}: {e}")
            return False
        if index.get("version") != _FORMAT_VERSION:
            return False
        tensors = _tensor_state(model)
        modules = dict(model.named_modules(remove_duplicate=False))
        with open(self.path / _DATA_FILE, "rb") as f, mmap.mmap(
                f.fileno(), 0, access=mmap.ACCESS_COPY) as data:
            restored: dict[str, torch.Tensor] = {}

            def load(entry: dict) -> torch.Tensor:
                start = entry["data_offset"]
                host = torch.frombuffer(data,
                                        dtype=torch.uint8,
                                        count=entry["storage_bytes"],
                                        offset=start)
                device = torch.device(entry["device"])
                storage = torch.empty(entry["storage_bytes"],
                                      dtype=torch.uint8,
                                      device=device)
                storage.copy_(host)
                return storage.view(_DTYPES[entry["dtype"]]).as_strided(
                    entry["shape"], entry["stride"], entry["storage_offset"])

            def view(entry: dict) -> torch.Tensor:
                base = restored[entry["view_of"]]
                dtype = _DTYPES[entry["dtype"]]
                flat = torch.empty(0, dtype=dtype, device=base.device)
                flat.set_(base.untyped_storage())
                return flat.as_strided(entry["shape"], entry["stride"],
                                       entry["byte_offset"] //
                                       flat.element_size())

            for name, entry in index["tensors"].items():
                module_name, _, attr = name.rpartition(".")
                module = modules[module_name]
                current = tensors.get(name)
                if (current is not None and not entry["replaced"]
                        and "view_of" not in entry
                        and self._fits(current, entry)):
                    # Keep the constructed tensor object (other modules may
                    # hold references to it) and fill it from the file.
                    current.data.view(-1).view(torch.uint8).copy_(
                        torch.frombuffer(data,
                                         dtype=torch.uint8,
                                         count=entry["storage_bytes"],
                                         offset=entry["data_offset"]))
                    restored[name] = current
                    continue
                tensor = view(entry) if "view_of" in entry else load(entry)
                restored[name] = tensor
                if entry["parameter"]:
                    module._parameters[attr] = nn.Parameter(
                        tensor, requires_grad=entry["requires_grad"])
                else:
                    module._buffers[attr] = tensor
            for name in index["removed_tensors"]:
                module_name, _, attr = name.rpartition(".")
                modules[module_name]._parameters.pop(attr, None)
                modules[module_name]._buffers.pop(attr, None)

            for module_name, attrs in index["attributes"].items():
                module = modules[module_name]
                for attr, entry in attrs["changed"].items():
                    if "value" in entry:
                        value = _decode_plain(entry["value"])
                    elif "view_of" in entry:
                        value = view(entry)
                    else:
                        value = load(entry)
                        restored[entry["tensor"]] = value
                    object.__setattr__(module, attr, value)
                for attr in attrs["removed"]:
                    if attr in vars(module):
                        delattr(module, attr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(f"Weights restored from the snapshot {self.path}.")
        return True
