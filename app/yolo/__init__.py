from .detection import process_frames, load_model
from .cuda import CUDA_ENABLED

__all__ = ["CUDA_ENABLED", "process_frames", "load_model"]
