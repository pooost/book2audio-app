"""Compute device selection and lightweight probing (no model loading)."""

from dataclasses import dataclass


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DeviceInfo:
    cuda_available: bool
    mps_available: bool
    cuda_device_name: str | None = None
    cuda_vram_gb: float | None = None
    selected: str = "cpu"


def probe_devices(requested: str = "auto") -> DeviceInfo:
    import torch

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    info = DeviceInfo(
        cuda_available=cuda_available,
        mps_available=mps_available,
        selected=pick_device(requested),
    )
    if cuda_available:
        info.cuda_device_name = torch.cuda.get_device_name(0)
        info.cuda_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    return info
