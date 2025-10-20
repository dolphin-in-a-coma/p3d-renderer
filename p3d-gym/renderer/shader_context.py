import numpy as np

from typing import Literal

from panda3d.core import Shader, Texture
from direct.showbase.ShowBase import ShowBase
from abc import ABC, abstractmethod


class P3DShaderContext(ABC):
    def __init__(self, 
                showbase: ShowBase,
                backend: Literal["instanced", "loop"] = "instanced" # TODO: Add class var for backend
        ) -> None:

        self.base = showbase
        if backend == "loop":
            raise NotImplementedError("Loop backend is not implemented yet. Use instanced backend instead.")
        self.backend = backend

        # (Registries are lazily created by child _register_self implementations.)
    
    @staticmethod
    def _make_shader() -> Shader:
        # TODO: it may be not static later.
        from pathlib import Path
        shader_dir = Path(__file__).resolve().parent.parent / 'shaders'
        vert_src = (shader_dir / 'basic.vert').read_text()
        frag_src = (shader_dir / 'basic.frag').read_text()
        return Shader.make(Shader.SL_GLSL, vert_src, frag_src)
    
    @staticmethod
    def _setup_buffer_texture(name: str, texels: int):
        """Setup a buffer texture for the shader."""
        # TODO: It's for RAM usage. To implement CUDA way too.
        from panda3d.core import GeomEnums
        tex = Texture(name)
        tex.setupBufferTexture(int(texels), Texture.T_float, Texture.F_rgba32, GeomEnums.UH_dynamic)
        tex.set_keep_ram_image(True)
        return tex

    @staticmethod
    def _pack_columns(mat_batch: np.ndarray) -> np.ndarray:
        # (B,4,4) -> (B,4,4) column-packed (transpose last two axes)
        return mat_batch.transpose(0, 2, 1).astype(np.float32, copy=False)

    @staticmethod
    def _rotation_mats_from_hpr(hpr_b3: np.ndarray) -> np.ndarray:
        """Vectorized rotation matrices from Euler angles (H, P, R) in radians.
        Order: R = Rz(H) @ Ry(P) @ Rx(R). Returns shape (B, 3, 3).
        """
        h = hpr_b3[:, 0]
        p = hpr_b3[:, 1]
        r = hpr_b3[:, 2]
        ch, sh = np.cos(h), np.sin(h)
        cp, sp = np.cos(p), np.sin(p)
        cr, sr = np.cos(r), np.sin(r)

        # Rz(H)
        Rz = np.zeros((hpr_b3.shape[0], 3, 3), dtype=np.float32)
        Rz[:, 0, 0] = ch; Rz[:, 0, 1] = -sh; Rz[:, 0, 2] = 0.0
        Rz[:, 1, 0] = sh; Rz[:, 1, 1] =  ch; Rz[:, 1, 2] = 0.0
        Rz[:, 2, 0] = 0.0; Rz[:, 2, 1] = 0.0; Rz[:, 2, 2] = 1.0

        # Ry(P)
        Ry = np.zeros((hpr_b3.shape[0], 3, 3), dtype=np.float32)
        Ry[:, 0, 0] =  cp; Ry[:, 0, 1] = 0.0; Ry[:, 0, 2] = sp
        Ry[:, 1, 0] = 0.0; Ry[:, 1, 1] = 1.0; Ry[:, 1, 2] = 0.0
        Ry[:, 2, 0] = -sp; Ry[:, 2, 1] = 0.0; Ry[:, 2, 2] = cp

        # Rx(R)
        Rx = np.zeros((hpr_b3.shape[0], 3, 3), dtype=np.float32)
        Rx[:, 0, 0] = 1.0; Rx[:, 0, 1] = 0.0; Rx[:, 0, 2] = 0.0
        Rx[:, 1, 0] = 0.0; Rx[:, 1, 1] =  cr; Rx[:, 1, 2] = -sr
        Rx[:, 2, 0] = 0.0; Rx[:, 2, 1] =  sr; Rx[:, 2, 2] =  cr

        # R = Rz @ Ry @ Rx
        Rzy = np.einsum('bij,bjk->bik', Rz, Ry)
        R = np.einsum('bij,bjk->bik', Rzy, Rx)
        return R.astype(np.float32, copy=False)


    def _set_shader_input(self, input_name: str, value) -> None:
        """Set a shader input parameter. If this object has an np, set it there; otherwise, broadcast to all registered P3DNodes on ShowBase."""
        np_model = getattr(self, 'np', None)
        if np_model is not None:
            np_model.setShaderInput(input_name, value)
            return
        for node_obj in getattr(self.base, '_p3d_nodes', []):
            np_node = getattr(node_obj, 'np', None)
            if np_node is None:
                continue
            np_node.setShaderInput(input_name, value)

    def _auto_screen_size_input(self) -> None:
        # TODO: change name to _auto_screen_size, and implement manual way to set screen size
        sx, sy = float(self.base.win.getXSize()), float(self.base.win.getYSize())
        self._set_shader_input('screenSize', (sx, sy))

    @abstractmethod
    def _register_self(self) -> None:
        """Child classes must register themselves on ShowBase (e.g., nodes list, camera singleton, or light singleton)."""
        raise NotImplementedError


