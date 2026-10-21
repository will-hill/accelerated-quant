"""accelerated_quant: import as aq."""
__version__ = "0.0.1"

_DEVICE = "cpu"


def device(name: str | None = None) -> str:
    """Select or report the array backend: "cpu" (NumPy) or "cuda" (CuPy)."""
    global _DEVICE
    if name is not None:
        if name not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        _DEVICE = name
    return _DEVICE

