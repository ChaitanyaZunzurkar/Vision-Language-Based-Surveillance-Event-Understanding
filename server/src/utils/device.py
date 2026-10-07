"""Central runtime device and CUDA diagnostics."""

from dataclasses import dataclass
from typing import Optional

from server.src.config.loader import config_loader
from server.src.utils.logger import logger


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    cuda_available: bool
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    torch_version: Optional[str]
    cuda_version: Optional[str]


def get_device_info() -> DeviceInfo:
    """Resolve the configured device once and fail clearly for invalid CUDA requests."""
    requested = str(config_loader.get("runtime.device", "auto")).lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("runtime.device must be one of: auto, cuda, cpu")

    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but PyTorch is not installed")
        return DeviceInfo("cpu", False, None, None, None, None)

    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    use_cuda = cuda_available and requested != "cpu" and bool(
        config_loader.get("runtime.use_cuda", True)
    )
    device = f"cuda:{int(config_loader.get('runtime.gpu_id', 0))}" if use_cuda else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else None
    memory = None
    if use_cuda:
        memory = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return DeviceInfo(
        device=device,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_memory_gb=memory,
        torch_version=getattr(torch, "__version__", None),
        cuda_version=getattr(torch.version, "cuda", None),
    )


DEVICE_INFO = get_device_info()


def log_device_diagnostics() -> DeviceInfo:
    info = DEVICE_INFO
    logger.info(
        "VISTA Runtime | device=%s cuda_available=%s gpu=%s vram_gb=%s "
        "torch=%s cuda_runtime=%s",
        info.device,
        info.cuda_available,
        info.gpu_name or "n/a",
        info.gpu_memory_gb or "n/a",
        info.torch_version or "n/a",
        info.cuda_version or "n/a",
    )
    return info
