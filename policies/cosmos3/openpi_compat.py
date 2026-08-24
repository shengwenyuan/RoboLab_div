"""Small OpenPI wire-protocol client used by the Cosmos 3 evaluator.

Keeping this code local avoids the ``openpi-client`` package's obsolete
``numpy<2`` constraint and lets Isaac Sim 6.0.1 retain its native dependency
set. The on-wire format matches OpenPI's MsgPack NumPy protocol.
"""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import numpy as np
from PIL import Image
import websockets.sync.client


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def resize_with_pad(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize images without distortion and zero-pad to ``height x width``."""
    if images.shape[-3:-1] == (height, width):
        return images
    original_shape = images.shape
    flat_images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_one(Image.fromarray(image), height, width) for image in flat_images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_one(image: Image.Image, height: int, width: int) -> np.ndarray:
    current_width, current_height = image.size
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    padded = Image.new(resized.mode, (width, height), 0)
    padded.paste(resized, ((width - resized_width) // 2, (height - resized_height) // 2))
    return np.asarray(padded)


class WebsocketClientPolicy:
    """Synchronous client for the OpenPI-compatible Cosmos policy server."""

    def __init__(self, host: str = "0.0.0.0", port: int | None = None) -> None:
        uri = host if host.startswith("ws") else f"ws://{host}"
        self._uri = f"{uri}:{port}" if port is not None else uri
        self._packer = _Packer()
        self._ws = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
        self._server_metadata = _unpackb(self._ws.recv())

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def infer(self, observation: dict) -> dict:
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return _unpackb(response)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self._ws.close()
