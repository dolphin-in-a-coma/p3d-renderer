import math
import re
from turtle import update
from typing import Literal
import numpy as np
import torch

from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, Texture
from direct.task import Task
from direct.showbase.ShowBaseGlobal import globalClock



from .node import P3DNode
from .camera import P3DCam
from .light import P3DLight
from .frame_grabber import GPUFrameGrabber, CPUFrameGrabber, GPU_AVAILABLE
# Support both package and script-style imports
try:
    from ..config import P3DConfig  # when imported as part of the package
except Exception:
    try:
        # when run with cwd at the package root (so `config.py` is top-level)
        from config import P3DConfig
    except Exception:
        # final fallback: add parent directory of this file to sys.path
        import os, sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from config import P3DConfig


class P3DRenderer(ShowBase):
    """ShowBase subclass that embeds scene helpers for multi-view rendering.

    This merges the responsibilities of `P3DScene` into a parent renderer so
    apps can simply inherit from `P3DRenderer` and call helper methods directly.
    """
    def __init__(self, cfg: P3DConfig | dict | None = None, **cfg_overrides):
        # Process config and apply PRC before initializing ShowBase
        self.cfg = P3DConfig.from_config(cfg, **cfg_overrides)
        prc_data = self.cfg.build_prc()
        loadPrcFileData('', prc_data)

        super().__init__()
        # Initial registries used by node/cam/light helpers
        self._p3d_nodes: list[P3DNode] = []
        self._p3d_cam: P3DCam | None = None
        self._p3d_light: P3DLight | None = None
        self.num_scenes = int(self.cfg.num_scenes)

        if self.cfg.report_fps:
            self._fps_last_print_time = 0.0 # TODO: move to state
            self.taskMgr.add(self.report_fps, 'report_fps')

        if self.cfg.manual_camera_control:
            self.enable_camera_control()

        # If running offscreen, disable the default main window's drawing to avoid extra work in PStats
        if self.cfg.offscreen and self.win is not None:
            try:
                for dr in list(self.win.getDisplayRegions()):
                    dr.setActive(False)
                self.win.setActive(False)
            except Exception:
                pass

        # self.taskMgr.doMethodLater(0, self._init_frame_grabber_once, 'init_frame_grabber_once')

    # --- Node helpers (mirroring P3DScene) ---
    def add_node(self,
            model_path: str | None,
            instances_per_scene: int,
            shared_across_scenes: bool = False,
            parent: 'P3DNode | None' = None,
            name: str | None = None,
            model_pivot_relative_point: tuple[float, float, float] | None = None,
            model_scale: tuple[float, float, float] = (10.0, 1.0, 1.0),
            positions: 'np.ndarray | None' = None,
            hprs: 'np.ndarray | None' = None,
            scales: 'np.ndarray | None' = None,
            colors: 'np.ndarray | None' = None,
            backend: Literal[ "loop", "instanced"] = "instanced",) -> P3DNode:
        node = P3DNode(self, model_path=model_path, num_scenes=self.num_scenes, instances_per_scene=int(instances_per_scene), shared_across_scenes=shared_across_scenes, parent=parent, name=name,
                       model_pivot_relative_point=model_pivot_relative_point, model_scale=model_scale, positions=positions, hprs=hprs, scales=scales, colors=colors, backend=backend)
        return node

    # --- Camera helpers ---
    @property
    def cam_manager(self) -> P3DCam | None:
        return getattr(self, '_p3d_cam', None)

    def add_camera(self) -> P3DCam:
        self._p3d_cam = P3DCam(self, num_scenes=self.cfg.num_scenes, cols=self.cfg.tiles[0], rows=self.cfg.tiles[1])
        return self._p3d_cam

    def _set_tiles_auto(self) -> None:
        if self._p3d_cam is None:
            self.add_camera()
        self._p3d_cam._set_tiles()

    def start(self) -> None:
        # TODO: to remove
        if self._p3d_cam is None:
            self.add_camera()
        self._p3d_cam.start_tasks(self.taskMgr)

    # --- Lighting helpers ---
    def add_light(self,
                  ambient: tuple[float, float, float] = (0.2, 0.2, 0.25),
                  dir_dir: tuple[float, float, float] = (0.4, -0.6, -0.7),
                  dir_col: tuple[float, float, float] = (1.0, 1.0, 1.0),
                  strength: float = 1.0) -> P3DLight:
        self._p3d_light = P3DLight(self, ambient=ambient, dir_dir=dir_dir, dir_col=dir_col, strength=strength)
        return self._p3d_light

    def set_key(self, key, value):
        self.keys[key] = value


    def enable_camera_control(self):
        if self.cfg.offscreen:
            print('WARNING: Manual camera control is not possible in offscreen mode, disabling.')
            self.cfg.manual_camera_control = False
            return

        self.taskMgr.add(self.update_camera, 'camera_control')

        self.keys = {
            "w": False, "s": False, "a": False, "d": False
        }
        self.accept("w", self.set_key, ["w", True])
        self.accept("s", self.set_key, ["s", True])
        self.accept("a", self.set_key, ["a", True])
        self.accept("d", self.set_key, ["d", True])
        self.accept("w-up", self.set_key, ["w", False])
        self.accept("s-up", self.set_key, ["s", False])
        self.accept("a-up", self.set_key, ["a", False])
        self.accept("d-up", self.set_key, ["d", False])


    def report_fps(self, task: Task) -> None:
        now = globalClock.getRealTime()
        if now - self._fps_last_print_time >= self.cfg.report_fps_interval:
            self._fps_last_print_time = now
            fps = globalClock.getAverageFrameRate() * self.cfg.num_scenes
            print(f"FPS: {float(fps):.1f}")

        return task.cont

    def update_camera(self, task):
        dt = self.clock.get_dt()
        speed = 25

        position_k3 = self._p3d_cam.get_eye()
        if self.keys["w"]:
            position_k3[:, 1] += speed * dt
        if self.keys["s"]:
            position_k3[:,1] -= speed * dt
        if self.keys["a"]:
            position_k3[:,0] -= speed * dt
        if self.keys["d"]:
            position_k3[:,0] += speed * dt
        self._p3d_cam.set_eye(position_k3)

        if self.mouseWatcherNode.hasMouse():
            md = self.win.getPointer(0)
            x = md.getX()
            y = md.getY()
            
            fwd_k3 = self._p3d_cam.get_forward()
            fwd_x, fwd_y, fwd_z = fwd_k3[:,0], fwd_k3[:,1], fwd_k3[:,2]

            self.win.movePointer(0, self.win.getXSize() // 2, self.win.getYSize() // 2)

        return task.cont

    def _setup_offscreen_rt(self):
        # Create an offscreen buffer sized to the tiled grid (cols x rows)
        cols = self.cfg.tiles[0]
        rows = self.cfg.tiles[1]
        tile_w = self.cfg.tile_resolution[0]
        tile_h = self.cfg.tile_resolution[1]
        w = max(1, cols * tile_w)
        h = max(1, rows * tile_h)
        self.offscreen_buffer = self.win.makeTextureBuffer('p3d-screen-rt', w, h)
        self.offscreen_tex = self.offscreen_buffer.getTexture()
        if self.offscreen_tex is not None:
            self.offscreen_tex.setKeepRamImage(True)
            self.offscreen_tex.setCompression(Texture.CMOff)

        # Ensure there is a camera rendering into the offscreen buffer
        # so the texture actually contains the rendered scene.
        try:
            if not hasattr(self, '_offscreen_cam_np') or self._offscreen_cam_np is None:
                # Create a camera for the offscreen buffer that mirrors the base camera
                self._offscreen_cam_np = self.makeCamera(self.offscreen_buffer)
                # Inherit transforms from the main camera so poses match
                try:
                    self._offscreen_cam_np.reparentTo(self.cam)
                except Exception:
                    pass
                # Match the main camera lens
                try:
                    self._offscreen_cam_np.node().setLens(self.cam.node().getLens())
                except Exception:
                    pass
        except Exception:
            pass

    def _warmup(self):
        # TODO: check if this is needed
        for _ in range(self.cfg.warmup_steps):
            self.taskMgr.step()

    def _init_frame_grabber_once(self, task):

        # self._warmup()
                    
        # Optional GPU grabber (CuPy + CUDA)
        # if getattr(self, 'offscreen_tex', None) is None:
        #     self._setup_offscreen_rt()

        print("GPU_AVAILABLE:", GPU_AVAILABLE)
        print("cuda_gl_interop:", self.cfg.cuda_gl_interop)
        print("device:", self.cfg.device)
    
        self._frame_grabber: GPUFrameGrabber | CPUFrameGrabber | None = None
        if GPU_AVAILABLE and self.cfg.cuda_gl_interop and self.cfg.device == 'cuda':
            try:
                self._frame_grabber = GPUFrameGrabber(self, self.offscreen_tex)
                print("GPU grabber initialized")
                return task.done
            except Exception as e:
                raise e # HACK: remove
                # self._gpu_grabber = None
    
        # Always have a CPU fallback grabber available
        try:
            self._frame_grabber = CPUFrameGrabber(self, getattr(self, 'offscreen_tex', None))
        except Exception:
            self._frame_grabber = None
        print("Frame grabber initialized:", self._frame_grabber)
        return task.done

    def grab_pixels(self):
        # Read pixels from the offscreen texture; rendering happens via taskMgr
        # Prefer GPU interop if available (returns torch CUDA tensor when obs_on_gpu=True)
        # if getattr(self, '_frame_grabber', None) is None:
        #     self.taskMgr.doMethodLater(0, self._init_frame_grabber_once, 'init_frame_grabber_once')
        #     return torch.zeros((int(self.cfg.window_resolution[1]), int(self.cfg.window_resolution[0]), self.cfg.num_channels), dtype=torch.uint8, device=self.cfg.device)
        try:
            frame = self._frame_grabber.grab()  # torch.uint8 [H,W,4] on CUDA
            frame = frame[..., :self.cfg.num_channels]
            return frame
        except Exception as e:
            if getattr(self, '_frame_grabber', None) is None:
                self.taskMgr.doMethodLater(0, self._init_frame_grabber_once, 'init_frame_grabber_once')
                return torch.zeros((int(self.cfg.window_resolution[1]), int(self.cfg.window_resolution[0]), self.cfg.num_channels), dtype=torch.uint8, device=self.cfg.device)
            
            raise e
            # return torch.zeros((int(self.cfg.window_resolution[1]), int(self.cfg.window_resolution[0]), self.cfg.num_channels), dtype=torch.uint8, device=self.cfg.device)

        # # CPU path: Read pixels from the offscreen texture (uint8 HxWx3) -> torch CPU tensor
        # tex = getattr(self, 'offscreen_tex', None)
        # if tex is None:
        #     import torch as _torch
        #     return _torch.zeros((int(self.cfg.height), int(self.cfg.width), 3), dtype=_torch.uint8)
        # # Pull latest GPU texture into RAM on every read
        # try:
        #     self.graphicsEngine.extractTextureData(tex, self.win.getGsg())
        # except Exception:
        #     pass
        # # Determine actual texture size
        # w = int(tex.getXSize()) or int(self.cfg.width)
        # h = int(tex.getYSize()) or int(self.cfg.height)
        # # Fetch raw bytes
        # try:
        #     data = tex.getRamImageAs('RGB')
        # except Exception:
        #     data = tex.getRamImage()
        # arr = np.frombuffer(data, dtype=np.uint8)
        # num_pixels = max(1, w * h)
        # channels = arr.size // num_pixels if num_pixels else 0
        # if channels and channels != 3:
        #     arr = arr.reshape((h, w, channels))[..., :3]
        # else:
        #     arr = arr.reshape((h, w, 3)) if arr.size >= w * h * 3 else np.zeros((h, w, 3), dtype=np.uint8)
        # # Flip vertically, ensure contiguous, and convert to torch CPU tensor
        # arr = np.flipud(arr).copy()
        # import torch as _torch
        # return _torch.from_numpy(arr)
    
    def step_and_grab(self):
        self.taskMgr.step()
        return self.grab_pixels()