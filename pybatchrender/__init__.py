from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("pybatchrender")
except Exception:
    __version__ = "0.1.0"

from .config import PBRConfig  # noqa: F401

# TorchRL is an optional dependency; guard the import so top-level package import works
try:
    from .env import PBREnv  # noqa: F401
except Exception:
    PBREnv = None  # type: ignore

try:
    from .renderer.renderer import PBRRenderer  # noqa: F401
except Exception:
    PBRRenderer = None  # type: ignore

# Expose renderer helpers (optional: depend on Panda3D/OpenGL availability)
try:
    from .renderer.node import PBRNode  # noqa: F401
except Exception:
    PBRNode = None  # type: ignore

try:
    from .renderer.camera import PBRCam  # noqa: F401
except Exception:
    PBRCam = None  # type: ignore

try:
    from .renderer.light import PBRLight  # noqa: F401
except Exception:
    PBRLight = None  # type: ignore

try:
    from .renderer.shader_context import PBRShaderContext  # noqa: F401
except Exception:
    PBRShaderContext = None  # type: ignore

try:
    from .renderer.frame_grabber import GPUFrameGrabber, CPUFrameGrabber, GPU_AVAILABLE  # noqa: F401
    # Convenience alias: prefer GPU if available
    FrameGrabber = GPUFrameGrabber if GPU_AVAILABLE else CPUFrameGrabber  # type: ignore
except Exception:
    GPUFrameGrabber = None  # type: ignore
    CPUFrameGrabber = None  # type: ignore
    GPU_AVAILABLE = False  # type: ignore
    FrameGrabber = None  # type: ignore

__all__ = [
    "PBRConfig",
    "PBREnv",
    "PBRRenderer",
    # Renderer helpers
    "PBRNode",
    "PBRCam",
    "PBRLight",
    "PBRShaderContext",
    "GPUFrameGrabber",
    "CPUFrameGrabber",
    "FrameGrabber",
    "__version__",
]
