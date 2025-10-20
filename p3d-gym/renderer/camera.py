import math
import numpy as np

from typing import Literal

from .shader_context import P3DShaderContext
from .node import P3DNode


class P3DCam(P3DShaderContext):
    def __init__(self, showbase, 
                num_scenes: int,
                cols: int | None = None,
                rows: int | None = None,
                backend: Literal[ "loop", "instanced"] = "instanced",
                fov_y_deg: float = 55.0,
                z_near: float = 0.05,
                z_far: float = 100.0,
                auto_tiles: bool = True,
                fixed_projection: bool = True, # if True, the projection will not be updated
                ) -> None:
        super().__init__(showbase, backend=backend)
        self.num_scenes = int(num_scenes)
        self.fov_y_deg = float(fov_y_deg)
        self.z_near = float(z_near)
        self.z_far = float(z_far)
        self.fixed_projection = fixed_projection
        # Create view/tile buffer textures
        self.viewbuf = self._setup_buffer_texture('p3d_cam_viewbuf', max(1,self.num_scenes)*4)
        self.tilebuf = self._setup_buffer_texture('p3d_cam_tilebuf', max(1,self.num_scenes))
        
        self.cols = cols
        self.rows = rows
        self._set_tiles()

        # Persistent camera state and VP cache
        # TODO: check these guys
        
        self.eye_k3: np.ndarray = np.zeros((self.num_scenes, 3), np.float32)
        self.forward_k3: np.ndarray = np.zeros((self.num_scenes, 3), np.float32)
        self.up_k3: np.ndarray = np.zeros((self.num_scenes, 3), np.float32)

        self.hpr_k3: np.ndarray = np.zeros((self.num_scenes, 3), np.float32) # ?
        self.target_k3: np.ndarray = np.zeros((self.num_scenes, 3), np.float32) # ?
        
        self.V_k44: np.ndarray = np.eye(4, 4, dtype=np.float32)[None, :, :].repeat(self.num_scenes, axis=0) # ?
        # self.P_k44: np.ndarray = np.zeros((self.num_scenes, 4, 4), np.float32) # ?

        self.VP_k44: np.ndarray = np.zeros((self.num_scenes, 4, 4), np.float32)

        # Projection cache
        self._proj_cache_key: tuple[float, float, float, float] | None = None
        self._proj_cache: np.ndarray | None = None
        self.P_k44: np.ndarray = self._get_projection()

        self._update_view(eye_k3=np.array([ 0., -12., 0.  ], np.float32), 
            forward_k3=np.array([0.0,  1. , 0.], np.float32),
            up_k3=np.array([0.0,  0.0,  1.], np.float32))

        self._update_vp()

        # Control whether to sync VP from Panda3D base camera each frame
        self.sync_from_base_cam: bool = False
        if hasattr(self.base, '_p3d_nodes') and self.base._p3d_nodes: # TODO: rename
            self.attach_all()
        # Register camera singleton
        self._register_self()

    def attach(self, node: P3DNode) -> None:
        # NOTE: P3DNode is not imported, so string typing, maybe change?
        node._set_shader_input('viewbuf', self.viewbuf)
        node._set_shader_input('tilebuf', self.tilebuf)
        node._set_shader_input('K', self.num_scenes)
        node._auto_screen_size_input()

    def attach_many(self, nodes: list[P3DNode]) -> None:
        for n in nodes:
            self.attach(n)

    def attach_all(self) -> None:
        self._set_shader_input('viewbuf', self.viewbuf)
        self._set_shader_input('tilebuf', self.tilebuf)
        self._set_shader_input('K', self.num_scenes)
        self._auto_screen_size_input()

    def _set_tiles_from_array(self, tiles_k4: np.ndarray) -> None:
        self.rows = tiles_k4.shape[0]
        self.cols = tiles_k4.shape[1]
        self.num_scenes = self.cols * self.rows
        self.tilebuf.set_ram_image(tiles_k4.astype(np.float32, copy=False).tobytes(order='C'))
        # TODO: add tiling based on number rows and cols

    # def _set_tiles_from_rows_cols(self, rows: int | None = None, cols: int | None = None) -> None:
    #     if rows is not None:
    #         self.rows = rows
    #     if cols is not None:
    #         self.cols = cols
    #     self.num_scenes = self.cols * self.rows
    #     tiles = np.zeros((self.num_scenes, 4), dtype=np.float32)
    #     for i in range(self.num_scenes):
    #         col = i % self.cols
    #         row = i // self.cols
    #         tiles[i] = [
    #             col / self.cols, (col + 1) / self.cols,
    #             1.0 - (row + 1) / self.rows, 1.0 - row / self.rows,
    #         ]
    #     self._set_tiles(tiles)

    def _set_tiles(self) -> None:
        if self.num_scenes is None and (self.cols is None or self.rows is None):
            raise ValueError("num_scenes or (cols and rows) must be provided")

        if self.cols is None and self.rows is None:
            self.cols = math.ceil(math.sqrt(self.num_scenes))

        if self.rows is None:
            self.rows = math.ceil(self.num_scenes / self.cols)

        if self.num_scenes is None:
            self.num_scenes = self.cols * self.rows
        
        tiles = np.zeros((self.num_scenes, 4), dtype=np.float32)
        for i in range(self.num_scenes):
            col = i % self.cols
            row = i // self.cols
            tiles[i] = [
                col / self.cols, (col + 1) / self.cols,
                1.0 - (row + 1) / self.rows, 1.0 - row / self.rows,
            ]
        self.tilebuf.set_ram_image(tiles.astype(np.float32, copy=False).tobytes(order='C'))

    # def _set_tiles_auto(self) -> None:
    #     # NOTE: this functional is repeated in config
    #     K = self.num_scenes
    #     cols = math.ceil(math.sqrt(K))
    #     rows = math.ceil(K / cols)
    #     self.cols, self.rows = cols, rows
    #     print(self.cols, self.rows)
    #     self._set_tiles_from_rows_cols()

    def _upload_viewproj(self, VP_k44: np.ndarray) -> None:
        self.viewbuf.set_ram_image(type(self)._pack_columns(VP_k44).tobytes(order='C'))

    def _get_projection(self) -> np.ndarray:
        # TODO: change aspect given tiled rendering
        if self.fixed_projection and self._proj_cache is not None:
            return self._proj_cache
        aspect = max(1e-6, float(self.base.win.getXSize()) / max(1, self.base.win.getYSize()))
        key = (float(self.fov_y_deg), float(self.z_near), float(self.z_far), float(aspect))
        if self._proj_cache_key != key or self._proj_cache is None:
            f = 1.0 / math.tan(math.radians(self.fov_y_deg) * 0.5)
            P = np.zeros((4, 4), np.float32)
            P[0, 0] = f / aspect
            P[1, 1] = f
            P[2, 2] = (self.z_far + self.z_near) / (self.z_near - self.z_far)
            P[2, 3] = (2.0 * self.z_far * self.z_near) / (self.z_near - self.z_far)
            P[3, 2] = -1.0
            self._proj_cache_key = key
            self._proj_cache = P
        return self._proj_cache

    def set_projection(self, fov_y_deg: float | None = None, z_near: float | None = None, z_far: float | None = None) -> None:
        # TODO: To check with P_k44 matrix
        # TODO: consider using aspect too
        if fov_y_deg is not None:
            self.fov_y_deg = float(fov_y_deg)
        if z_near is not None:
            self.z_near = float(z_near)
        if z_far is not None:
            self.z_far = float(z_far)
        # Invalidate projection cache and re-upload with current VP
        self._proj_cache_key = None
        self._proj_cache = None
        if self.VP_k44.size:
            self._upload_viewproj(self.VP_k44)

    def _ensure_kx3(self, arr: np.ndarray | list | tuple, name: str) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (K,3) or (3,), got {a.shape}")
        if a.shape[0] == 1 and self.num_scenes > 1:
            a = np.repeat(a, self.num_scenes, axis=0)
        if a.shape[0] != self.num_scenes:
            raise ValueError(f"{name} first dim must be K={self.num_scenes}, got {a.shape[0]}")
        return a

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.where(n > 1e-8, n, 1.0)
    # TODO: move normalize to proper place

    def _update_view(self, eye_k3: np.ndarray | None = None, 
                    forward_k3: np.ndarray | None = None,
                    up_k3: np.ndarray | None = None) -> np.ndarray:

        basis_changed = (forward_k3 is not None) or (up_k3 is not None)
        eye_changed = (eye_k3 is not None)

        if not basis_changed and not eye_changed:
            return

        if basis_changed:
            if forward_k3 is not None:
                forward_k3 = self._ensure_kx3(forward_k3, 'forward_k3')
                f = type(self)._normalize(forward_k3)
            else:
                f = -self.V_k44[:, 2, 0:3]
            if up_k3 is not None:
                up_k3 = self._ensure_kx3(up_k3, 'up_k3')
                up_in = type(self)._normalize(up_k3)
            else:
                up_in = self.V_k44[:, 1, 0:3]
            s = type(self)._normalize(np.cross(f, up_in))
            u = np.cross(s, f)

            self.V_k44[:, 0, 0:3] = s
            self.V_k44[:, 1, 0:3] = u
            self.V_k44[:, 2, 0:3] = -f
        else:
            s = self.V_k44[:, 0, 0:3]
            u = self.V_k44[:, 1, 0:3]
            f = -self.V_k44[:, 2, 0:3]


        if basis_changed or eye_changed:
            if eye_changed: # NOTE: I don't use _ensure_kx3 here, maybe should?
                eye_k3 = self._ensure_kx3(eye_k3, 'eye_k3')
                self.eye_k3 = eye_k3
            self.V_k44[:, 0, 3] = -np.einsum('ij,ij->i', s, self.eye_k3)
            self.V_k44[:, 1, 3] = -np.einsum('ij,ij->i', u, self.eye_k3)
            self.V_k44[:, 2, 3] =  np.einsum('ij,ij->i', f, self.eye_k3)


    @staticmethod
    def _fwd_up_from_hpr(hpr_k3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R = P3DShaderContext._rotation_mats_from_hpr(hpr_k3)
        fwd = np.einsum('bij,j->bi', R, np.array([0.0, 1.0, 0.0], np.float32))
        up = np.einsum('bij,j->bi', R, np.array([0.0, 0.0, 1.0], np.float32))
        return fwd, up

    def _update_vp(self) -> None:
        # print(self.P_k44.shape, self.V_k44.shape)
        VP = self.P_k44 @ self.V_k44

        # print(self.V_k44[0])

        VP = VP.reshape(-1, 4, 4)
        # VP = np.repeat(VP, self.num_scenes, axis=0)

        self.VP_k44 = VP.astype(np.float32, copy=False)
        self._upload_viewproj(self.VP_k44)

    def get_vp(self) -> np.ndarray:
        return self.VP_k44.copy()

    # --- Simplified public API ---
    def look_at(self, target_k3: np.ndarray | list | tuple,
                lazy: bool = False) -> None:
        fwd = self._fwd_from_lookat(target_k3)
        self._update_view(forward_k3=fwd)
        if not lazy:
            self._update_vp()

    def set_positions(self, eye_k3: np.ndarray | list | tuple, keep_lookat: bool = False, lazy: bool = False) -> None:
        if keep_lookat:
            raise NotImplementedError('Target can\'t be reconsturcted from eye and forward, needs to be stored in memory then')
        else:
            self.set_eye(eye_k3, lazy=lazy)

    def set_positions_and_lookat(self, eye_k3: np.ndarray | list | tuple, 
                                 target_k3: np.ndarray | list | tuple, 
                                 lazy: bool = False) -> None:

        eye_k3 = self._ensure_kx3(eye_k3, 'eye_k3')
        target_k3 = self._ensure_kx3(target_k3, 'target_k3')

        fwd = self._fwd_from_lookat(target_k3, eye_k3)
        self._update_view(eye_k3=eye_k3, forward_k3=fwd)
        if not lazy:
            self._update_vp()

    def _fwd_from_lookat(self, target_k3: np.ndarray | list | tuple, eye_k3: np.ndarray | list | tuple | None = None) -> np.ndarray:
        eye_k3 = eye_k3 if eye_k3 is not None else self.eye_k3
        fwd = target_k3 - eye_k3
        return fwd

    def set_hprs(self, hpr_k3: np.ndarray | list | tuple, lazy: bool = False) -> None:
        hpr = self._ensure_kx3(hpr_k3, 'hpr_k3')
        fwd, up = self._fwd_up_from_hpr(hpr)
        self._update_view(forward_k3=fwd, up_k3=up)
        if not lazy:
            self._update_vp()

    # --- Axis setters/getters (use _update_view) ---
    def set_eye(self, eye_k3: np.ndarray | list | tuple, lazy: bool = False) -> None:
        self._update_view(eye_k3=eye_k3)
        if not lazy:
            self._update_vp()

    def set_forward(self, forward_k3: np.ndarray | list | tuple, lazy: bool = False) -> None:
        self._update_view(forward_k3=forward_k3)
        if not lazy:
            self._update_vp()

    def set_up(self, up_k3: np.ndarray | list | tuple, lazy: bool = False) -> None:
        self._update_view(up_k3=up_k3)
        if not lazy:
            self._update_vp()

    def set_right(self, right_k3: np.ndarray | list | tuple, lazy: bool = False) -> None:
        # Keep current forward, adjust up so that resulting right matches desired
        right_k3 = type(self)._normalize(right_k3)
        f = -self.V_k44[:, 2, 0:3]
        u = np.cross(right_k3, f)
        # Delegate to _update_view with fixed forward and derived up
        self._update_view(forward_k3=f, up_k3=u)
        if not lazy:
            self._update_vp()

    def get_eye(self) -> np.ndarray:
        return self.eye_k3.copy()

    def get_forward(self) -> np.ndarray:
        # Prefer cached forward if available; derive from V otherwise
        return (-self.V_k44[:, 2, 0:3]).copy()

    def get_up(self) -> np.ndarray:
        return self.V_k44[:, 1, 0:3].copy()

    def get_right(self) -> np.ndarray:
        return self.V_k44[:, 0, 0:3].copy()

    def _compute_vp_from_camera(self) -> np.ndarray:
        import math as _math
        from panda3d.core import Vec3
        # Camera world pose
        eye = self.base.cam.getPos(self.base.render)
        q = self.base.cam.getQuat(self.base.render)
        fwd = q.xform(Vec3(0, 1, 0))
        upw = q.xform(Vec3(0, 0, 1))
        target = eye + fwd

        def _normalize(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v)
            return v / (n if n > 1e-8 else 1.0)

        eye_np = np.array([eye.x, eye.y, eye.z], dtype=np.float32)
        f_np = _normalize(np.array([target.x, target.y, target.z], np.float32) - eye_np)
        up_np = _normalize(np.array([upw.x, upw.y, upw.z], np.float32))
        s = _normalize(np.cross(f_np, up_np))
        u = np.cross(s, f_np)

        V = np.zeros((4, 4), np.float32)
        V[0, 0:3] = s
        V[1, 0:3] = u
        V[2, 0:3] = -f_np
        V[3, 3] = 1.0
        V[0, 3] = -np.dot(s, eye_np)
        V[1, 3] = -np.dot(u, eye_np)
        V[2, 3] =  np.dot(f_np, eye_np)

        aspect = max(1e-6, float(self.base.win.getXSize()) / max(1, self.base.win.getYSize()))
        fov_y_deg = 55.0
        z_near, z_far = 0.05, 100.0
        f = 1.0 / _math.tan(_math.radians(fov_y_deg) * 0.5)
        P = np.zeros((4, 4), np.float32)
        P[0, 0] = f / aspect
        P[1, 1] = f
        P[2, 2] = (z_far + z_near) / (z_near - z_far)
        P[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
        P[3, 2] = -1.0

        # print(P.shape, V.shape)
        # print(f'{P[0]=}')
        # print(f'{V[0]=}')
        # print(f'{eye_np=}')
        # print(f'{f_np=}')
        # print(f'{up_np=}')
        # print(f'{s=}')
        # print(f'{u=}')
        # print(f'{f_np=}')
        # print(f'{up_np=}')
        # print(f'{s=}')
        # print(f'{u=}')
        # print(f'{P=}')
        # print(f'{V=}')

        VP = P @ V
        return np.stack([VP] * self.num_scenes, axis=0)

    def start_tasks(self, taskMgr, name: str = 'p3d_cam_update'):
        def _update(task):
            # Update VP only if syncing from Panda base camera is enabled
            if self.sync_from_base_cam:
                VP = self._compute_vp_from_camera()
                # print(VP)
                self._upload_viewproj(VP)
            # Keep screenSize current
            try:
                if hasattr(self.base, 'offscreen_buffer') and self.base.offscreen_buffer is not None:
                    sx, sy = float(self.base.offscreen_buffer.getXSize()), float(self.base.offscreen_buffer.getYSize())
                else:
                    sx, sy = float(self.base.win.getXSize()), float(self.base.win.getYSize())
                # Broadcast to all registered nodes
                self._set_shader_input('screenSize', (sx, sy))
            except Exception:
                pass
            return task.cont
        taskMgr.add(_update, name)

    def enable_base_cam_sync(self) -> None:
        self.sync_from_base_cam = True

    def disable_base_cam_sync(self) -> None:
        self.sync_from_base_cam = False

    def _register_self(self) -> None:
        # Camera singleton on ShowBase
        self.base._p3d_cam = self


