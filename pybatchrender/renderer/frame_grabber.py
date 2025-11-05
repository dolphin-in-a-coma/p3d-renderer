import torch
from abc import ABC, abstractmethod
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Texture

# Optional CUDA / CuPy / OpenGL imports for GPU grabbing (NVIDIA only)
try:
    from cuda import cudart  # type: ignore
    import cupy as cp  # type: ignore
    from OpenGL.GL import GL_TEXTURE_2D  # type: ignore
    GPU_AVAILABLE = True
    # torch = torch 
except Exception as e:
    print("Error importing CUDA/CuPy/OpenGL:", e)
    print("Cuda-OpenGL interop will not be used.")

    GPU_AVAILABLE = False
    cudart = None  # type: ignore
    cp = None  # type: ignore
    # torch = None

class BaseFrameGrabber(ABC):
    """Abstract base class for frame grabbers."""

    def __init__(self, base: ShowBase, tex: Texture | None):
        self.base = base
        self.tex = tex

    @abstractmethod
    def grab(self):
        raise NotImplementedError

    def close(self):
        pass


class GPUFrameGrabber(BaseFrameGrabber):
    """Zero-copy GPU-to-GPU frame grabber using CUDA–OpenGL interop.

    Returns a torch CUDA uint8 tensor shaped [H, W, 4] (RGBA), flipped vertically
    to match conventional image coords.
    """

    def __init__(self, base: ShowBase, tex: Texture, readonly: bool = True):
        if not GPU_AVAILABLE:
            raise RuntimeError("CUDA / OpenGL interop not available on this system.")
        super().__init__(base, tex)
        self.W, self.H = tex.getXSize(), tex.getYSize()

        # Ensure the texture is prepared and get native GL id
        gsg = base.win.getGsg()
        tctx = tex.prepareNow(0, gsg.getPreparedObjects(), gsg)
        tex_id = tctx.getNativeId()

        flags = (
            cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
            if readonly else 0
        )

        self.resource = type(self)._cuda_check(
            cudart.cudaGraphicsGLRegisterImage(tex_id, GL_TEXTURE_2D, flags)
        )

        # Destination CuPy buffer you can reuse
        self._cupy_buf = cp.empty((self.H, self.W, 4), dtype=cp.uint8)

    @staticmethod
    def _cuda_check(err_tuple):
        if not GPU_AVAILABLE:
            return None
        if isinstance(err_tuple, tuple):
            err, *rest = err_tuple
        else:
            err, rest = err_tuple, None
        # Import lazily to avoid unconditional dependency
        from cuda import cudart as _cudart  # type: ignore
        if err != _cudart.cudaError_t.cudaSuccess:  # type: ignore
            name = _cudart.cudaGetErrorName(err)[1].decode()  # type: ignore
            desc = _cudart.cudaGetErrorString(err)[1].decode()  # type: ignore
            raise RuntimeError(f"{name}({int(err)}): {desc}")
        if rest:
            return rest[0]
        return None

    def grab(self):

        type(self)._cuda_check(cudart.cudaGraphicsMapResources(1, self.resource, None))
        cuarray = type(self)._cuda_check(
            cudart.cudaGraphicsSubResourceGetMappedArray(self.resource, 0, 0)
        )
        type(self)._cuda_check(
            cudart.cudaMemcpy2DFromArray(
                self._cupy_buf.data.ptr,
                self.W * 4,
                cuarray,
                0,
                0,
                self.W * 4,
                self.H,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            )
        )
        type(self)._cuda_check(cudart.cudaGraphicsUnmapResources(1, self.resource, None))
        # Convert CuPy to torch without copy and flip vertically
        t = torch.utils.dlpack.from_dlpack(self._cupy_buf.toDlpack()).flip(0)
        return t

    def close(self):
        type(self)._cuda_check(cudart.cudaGraphicsUnregisterResource(self.resource))


class CPUFrameGrabber(BaseFrameGrabber):
    """CPU-based frame grabber that reads pixels from a Panda3D Texture.

    Returns a torch CPU uint8 tensor shaped [H, W, 3] (RGB), flipped vertically.
    Logic mirrors the existing CPU path used in renderer.grab_pixels.
    """

    def __init__(self, base: ShowBase, tex: Texture | None):
        super().__init__(base, tex)

    def grab(self):
        tex = self.tex
        # print("tex:", tex)
        if tex is None:
            # Use offscreen buffer size if available, otherwise fallback to window size
            try:
                if hasattr(self.base, 'offscreen_buffer') and self.base.offscreen_buffer is not None:
                    w = max(1, int(self.base.offscreen_buffer.getXSize()))
                    h = max(1, int(self.base.offscreen_buffer.getYSize()))
                else:
                    w = max(1, int(self.base.win.getXSize()))
                    h = max(1, int(self.base.win.getYSize()))
            except Exception:
                w = max(1, int(self.base.win.getXSize()))
                h = max(1, int(self.base.win.getYSize()))
            return torch.zeros((h, w, 3), dtype=torch.uint8)

        # Pull latest GPU texture into RAM on every read
        try:
            self.base.graphicsEngine.extractTextureData(tex, self.base.win.getGsg())
        except Exception:
            pass

        # Determine actual texture size
        w = int(tex.getXSize()) or (int(self.base.offscreen_buffer.getXSize()) if hasattr(self.base, 'offscreen_buffer') and self.base.offscreen_buffer is not None else int(self.base.win.getXSize()))
        h = int(tex.getYSize()) or (int(self.base.offscreen_buffer.getYSize()) if hasattr(self.base, 'offscreen_buffer') and self.base.offscreen_buffer is not None else int(self.base.win.getYSize()))

        # Fetch raw bytes
        try:
            data = tex.getRamImageAs('RGB')
        except Exception:
            data = tex.getRamImage()

        # Interpret bytes as uint8 tensor efficiently using ByteStorage
        storage = torch.ByteStorage.from_buffer(data)  # shares memory with underlying buffer
        flat = torch.ByteTensor(storage)
        num_pixels = max(1, w * h)
        channels = flat.numel() // num_pixels if num_pixels else 0
        if channels and channels >= 3:
            t = flat.view(h, w, channels)[..., :3].contiguous()
        else:
            # Fallback to zeros if data is insufficient
            t = torch.zeros((h, w, 3), dtype=torch.uint8)

        # Flip vertically and return
        return t.flip(0).contiguous()

    def close(self):
        pass
