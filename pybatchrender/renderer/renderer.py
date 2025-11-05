import math
import re
from typing import Literal
import torch

from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, Texture, MouseButton
from direct.task import Task
from direct.showbase.ShowBaseGlobal import globalClock



from .node import PBRNode
from .camera import PBRCam
from .light import PBRLight
from .frame_grabber import GPUFrameGrabber, CPUFrameGrabber, GPU_AVAILABLE
# Support both package and script-style imports
try:
    from ..config import PBRConfig  # when imported as part of the package
except Exception:
    try:
        # when run with cwd at the package root (so `config.py` is top-level)
        from config import PBRConfig
    except Exception:
        # final fallback: add parent directory of this file to sys.path
        import os, sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from config import PBRConfig


class PBRRenderer(ShowBase):
    """ShowBase subclass that embeds scene helpers for multi-view rendering.

    This merges the responsibilities of `PBRScene` into a parent renderer so
    apps can simply inherit from `PBRRenderer` and call helper methods directly.
    """
    def __init__(self, cfg: PBRConfig | dict | None = None, **cfg_overrides):
        
        # Process config and apply PRC before initializing ShowBase
        self.cfg = PBRConfig.from_config(cfg, **cfg_overrides)
        prc_data = self.cfg.build_prc()
        loadPrcFileData('', prc_data)
        super().__init__()

        # Print backend and configuration info
        gsg = self.win.getGsg()
        print("Pipe:", self.win.pipe.getType().getName()) # expect: eglGraphicsPipe
        print("Renderer:", gsg.getDriverRenderer()) # e.g., "NVIDIA L4/PCIe/SSE2"
        print("Config:", self.cfg)
        print("Prc data:", prc_data)

        # Initial registries used by node/cam/light helpers
        self._pbr_nodes: list[PBRNode] = []
        self._pbr_cam: PBRCam | None = None
        self._pbr_light: PBRLight | None = None
        self.num_scenes = int(self.cfg.num_scenes)

        # FPS reporting
        if self.cfg.report_fps:
            self._fps_last_print_time = 0.0 # TODO: move to state
            self.taskMgr.add(self.report_fps, 'report_fps')

        self.disableMouse()
        # Camera control with WASD and mouse 
        if self.cfg.manual_camera_control:
            self.enable_camera_control()

        # For offscreen, disable the default main window's drawing
        # TODO: double check this
        if self.cfg.offscreen and self.win is not None:
            try:
                for dr in list(self.win.getDisplayRegions()):
                    dr.setActive(False)
                self.win.setActive(False)
            except Exception:
                pass

    def add_node(self,
            model_path: str | None,
            instances_per_scene: int,
            texture: Texture | str | bool | None = None,
            model_pivot_relative_point: tuple[float, float, float] | None = None,
            model_scale: tuple[float, float, float] = (10.0, 1.0, 1.0),
            positions: 'torch.Tensor | None' = None,
            hprs: 'torch.Tensor | None' = None,
            scales: 'torch.Tensor | None' = None,
            colors: 'torch.Tensor | None' = None,
            backend: Literal[ "loop", "instanced"] = "instanced",
            shared_across_scenes: bool = False,
            parent: 'PBRNode | None' = None,
            name: str | None = None,
            ) -> PBRNode:
        node = PBRNode(self, model_path=model_path, 
                        num_scenes=self.num_scenes,
                        instances_per_scene=int(instances_per_scene), 
                        texture=texture,
                        model_pivot_relative_point=model_pivot_relative_point,
                        model_scale=model_scale,
                        positions=positions, 
                        hprs=hprs,
                        scales=scales,
                        colors=colors,
                        backend=backend,
                        shared_across_scenes=shared_across_scenes,
                        parent=parent,
                        name=name)
        return node

    def add_camera(self) -> PBRCam:
        self._pbr_cam = PBRCam(self, num_scenes=self.cfg.num_scenes, cols=self.cfg.tiles[0], rows=self.cfg.tiles[1])
        return self._pbr_cam

    # TODO: is this still of any use?
    def _set_tiles_auto(self) -> None:
        if self._pbr_cam is None:
            self.add_camera()
        self._pbr_cam._set_tiles()

    # --- Lighting helpers ---
    def add_light(self,
                  ambient: tuple[float, float, float] = (0.2, 0.2, 0.25),
                  dir_dir: tuple[float, float, float] = (0.4, -0.6, -0.7),
                  dir_col: tuple[float, float, float] = (1.0, 1.0, 1.0),
                  strength: float = 1.0) -> PBRLight:
        self._pbr_light = PBRLight(self, ambient=ambient, dir_dir=dir_dir, dir_col=dir_col, strength=strength)
        return self._pbr_light

    def set_key(self, key, value):
        self.keys[key] = value

    def enable_camera_control(self):
        if self.cfg.offscreen:
            print('WARNING: Manual camera control is not possible in offscreen mode, disabling.')
            self.cfg.manual_camera_control = False
            return

        self.keys = {
            "w": False, "s": False, "a": False, "d": False,
            "z": False, "x": False,
        }
        self.accept("w", self.set_key, ["w", True])
        self.accept("s", self.set_key, ["s", True])
        self.accept("a", self.set_key, ["a", True])
        self.accept("d", self.set_key, ["d", True])
        self.accept("w-up", self.set_key, ["w", False])
        self.accept("s-up", self.set_key, ["s", False])
        self.accept("a-up", self.set_key, ["a", False])
        self.accept("d-up", self.set_key, ["d", False])
        
        # up/down movement of the camera
        self.accept("z", self.set_key, ["z", True])
        self.accept("x", self.set_key, ["x", True])
        self.accept("z-up", self.set_key, ["z", False])
        self.accept("x-up", self.set_key, ["x", False])

        self._camera_speed = 25    

        self._prev_mouse_x = 0
        self._prev_mouse_y = 0
        self._mouse_sensitivity = 0.01


        self.taskMgr.add(self.update_camera, 'camera_control')

    def update_camera(self, task):
        dt = self.clock.get_dt()

        position_k3 = self._pbr_cam.get_eye()
        if self.keys["w"]:
            position_k3[:, 1] += self._camera_speed * dt
        if self.keys["s"]:
            position_k3[:,1] -= self._camera_speed * dt
        if self.keys["a"]:
            position_k3[:,0] -= self._camera_speed * dt
        if self.keys["d"]:
            position_k3[:,0] += self._camera_speed * dt
        if self.keys["z"]:
            position_k3[:,2] += self._camera_speed * dt
        if self.keys["x"]:
            position_k3[:,2] -= self._camera_speed * dt
        position_k3 = position_k3.reshape(-1, 3)
        self._pbr_cam.set_eye(position_k3)

        if self.mouseWatcherNode.hasMouse():
            md = self.win.getPointer(0)
            mouse_dx = md.getX() - self._prev_mouse_x
            mouse_dy = md.getY() - self._prev_mouse_y
            self._prev_mouse_x = md.getX()
            self._prev_mouse_y = md.getY()
            
            # Only rotate camera while left mouse button is held
            if self.mouseWatcherNode.is_button_down(MouseButton.one()):
                fwd_k3 = self._pbr_cam.get_forward()
                # TODO: implement with hpr
                fwd_x, fwd_y, fwd_z = fwd_k3[:,0], fwd_k3[:,1], fwd_k3[:,2]
                new_fwd = torch.stack([
                    fwd_x + float(mouse_dx) * self._mouse_sensitivity,
                    fwd_y,
                    fwd_z + float(mouse_dy) * self._mouse_sensitivity,
                ], dim=1)
                new_fwd = self._pbr_cam._normalize(new_fwd)
                self._pbr_cam.set_forward(new_fwd)

            # self.win.movePointer(0, int(self.win.getXSize() // 2), int(self.win.getYSize() // 2))

        return task.cont

    def report_fps(self, task: Task) -> None:
        now = globalClock.getRealTime()
        if now - self._fps_last_print_time >= self.cfg.report_fps_interval:
            self._fps_last_print_time = now
            fps = globalClock.getAverageFrameRate() * self.cfg.num_scenes
            print(f"FPS: {float(fps):.1f}")

        return task.cont

    def _setup_offscreen_rt(self):
        # Create an offscreen buffer sized to the tiled grid (cols x rows)
        cols = self.cfg.tiles[0]
        rows = self.cfg.tiles[1]
        tile_w = self.cfg.tile_resolution[0]
        tile_h = self.cfg.tile_resolution[1]
        w = max(1, cols * tile_w)
        h = max(1, rows * tile_h)
        self.offscreen_buffer = self.win.makeTextureBuffer('pbr-screen-rt', w, h)
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


                    
        # #Optional GPU grabber (CuPy + CUDA)
        # if getattr(self, 'offscreen_tex', None) is None:
        #     self._setup_offscreen_rt()

        # self._warmup()

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
        if getattr(self, '_frame_grabber', None) is None:
            # Try a synchronous CPU fallback to avoid zero frames on first grabs
            try:
                if not hasattr(self, 'offscreen_tex') or self.offscreen_tex is None:
                    self._setup_offscreen_rt()
                self._frame_grabber = CPUFrameGrabber(self, getattr(self, 'offscreen_tex', None))
                try:
                    self.taskMgr.step()
                except Exception:
                    pass
            except Exception:
                self.taskMgr.doMethodLater(0, self._init_frame_grabber_once, 'init_frame_grabber_once')
                return torch.zeros((int(self.cfg.window_resolution[1]), int(self.cfg.window_resolution[0]), self.cfg.num_channels), dtype=torch.uint8)

        try:
            frame = self._frame_grabber.grab()  # torch.uint8 [H,W,CH]
            frame = frame[..., :self.cfg.num_channels]
            return frame
        except Exception:
            # Schedule async init and return zeros this frame
            self.taskMgr.doMethodLater(0, self._init_frame_grabber_once, 'init_frame_grabber_once')
            return torch.zeros((int(self.cfg.window_resolution[1]), int(self.cfg.window_resolution[0]), self.cfg.num_channels), dtype=torch.uint8)
    
    def _rearrange_img(self, img: torch.Tensor) -> torch.Tensor:

        # print(img)
        # img: (R H)(CL W)CH
        # print(img.shape)
        img = img.permute(2, 0, 1) # -> CH(R H)(CL W)
        # print(img.shape)
        # img = img.contiguous() # seems to be unnecessary
        img = img.view(self.cfg.num_channels, self.cfg.tiles[1], self.cfg.tile_resolution[1], self.cfg.tiles[0], self.cfg.tile_resolution[0]) # -> CH R H CL W
        img = img.permute(1, 3, 0, 2, 4) # -> R CL CH H W
        img = img.reshape(-1, self.cfg.num_channels, self.cfg.tile_resolution[1], self.cfg.tile_resolution[0]) # -> B CH H W
        return img[:self.cfg.num_scenes]

    def _step(self, *args, **kwargs):
        # Default no-op. Children override this to update scene state per step.
        return None

    def _interactive_step(self):
        # Default: no-op. Children override to read inputs and call _step(...)
        return None

    def _interactive_step_with_task(self, task):
        self._interactive_step() # TODO: consider whether dancing around .taskMgr is necessary
        return task.cont

    def step(self, *args, return_pixels: bool = True, **kwargs):
        # In interactive mode, a task drives stepping. We only advance one frame here.
        if getattr(self.cfg, 'interactive', False):
            self.taskMgr.step()
        else:
            self._step(*args, **kwargs)
            self.taskMgr.step()
        if return_pixels:
            img = self.grab_pixels()
            # print(img.shape)
            img_processed = self._rearrange_img(img)
            # print(img_processed.shape)
            return img_processed


    def __call__(self, *args, return_pixels: bool = True, **kwargs):
        return self.step(*args, return_pixels=return_pixels, **kwargs)

    def _init_frame_grabber(self):
        self._setup_offscreen_rt()
        self._warmup()
        self.taskMgr.doMethodLater(0, self._init_frame_grabber_once, 'init-frame-grabber-once')

    def setup_environment(self) -> None:

        if self._pbr_cam is None:
            self.add_camera() # TODO: consider removing this
        if getattr(self.cfg, 'interactive', False):
            self.taskMgr.add(self._interactive_step_with_task, 'pbr-interactive-step')
        self._init_frame_grabber()
