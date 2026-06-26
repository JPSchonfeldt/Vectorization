"""
Stage C - rembg-based background removal.

Wraps rembg in a small session cache so the ONNX model only loads once per
container lifetime. First-call latency is high (model download or cold
start); subsequent calls are fast.
"""

from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    from rembg import new_session, remove
except Exception as ex:
    new_session = None
    remove = None
    _import_error = ex
else:
    _import_error = None


_session_cache: dict = {}


def get_session(model_name: str):
    """Return a cached rembg session for the given model, creating it once."""

    if new_session is None:
        raise Exception(
            "rembg is not available in this environment: "
            + (str(_import_error) if _import_error else "unknown error")
        )

    if model_name not in _session_cache:
        try:
            _session_cache[model_name] = new_session(model_name)
        except Exception as ex:
            raise Exception(f"Failed to load rembg model {model_name!r}: {ex}")

    return _session_cache[model_name]


def remove_background(
    image: Image.Image,
    model_name: str = "u2netp",
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """Run rembg on the supplied PIL image.

    Returns a tuple of:
      - RGB image (background pixels replaced)
      - uint8 alpha mask of shape (H, W), 255 = foreground and 0 = removed
    """

    if image.mode != "RGBA":
        prepared = image.convert("RGBA")
    else:
        prepared = image

    session = get_session(model_name)

    cutout = remove(prepared, session=session)

    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")

    rgba_array = np.array(cutout)
    alpha_mask = rgba_array[:, :, 3].astype(np.uint8)

    rgb_image = Image.fromarray(rgba_array[:, :, :3], mode="RGB")

    return rgb_image, alpha_mask


def clear_session_cache() -> None:
    """Clear cached rembg sessions. Mostly useful for tests."""

    _session_cache.clear()