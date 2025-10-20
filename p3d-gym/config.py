from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict

import math
from typing import Iterable


@dataclass
class P3DConfig:
    offscreen: bool = True

    num_scenes: int | None = None
    tiles: tuple[int, int] | int | None = None
    tile_resolution: tuple[int, int] | None = None
    window_resolution: tuple[int, int] | None = None

    num_channels: int = 3
    batch_inner_dim: int | None = None

    clip_camera: tuple[float, float] = (3.0, 500.0)
    min_objects: int = 50
    max_objects: int = 100
    device: str | None = None
    panda3d_backend: str | None = 'arm'
    log_level: int = logging.DEBUG
    extra_prc_file_data: str = (
        'audio-library-name null\n'
        'textures-power-2 none\n'
        'sync-video 0\n'
    )

    render_mode: str = 'rgb_array'
    dt: float = 1/60
    warm_up_steps: int = 5

    report_fps: bool = True
    report_fps_interval: float = 1.0

    manual_camera_control: bool = False

    def process_resolution(self) -> None:

        if self.tiles is None and self.num_scenes is not None:
            self.tiles = self.num_scenes

        if isinstance(self.tiles, int):
            cols = math.ceil(math.sqrt(self.tiles))
            rows = math.ceil(self.tiles / cols)
            self.tiles = (cols, rows)

        num_nones = sum(1 for x in [self.tiles, self.tile_resolution, self.window_resolution] if x is None)

        if num_nones == 0:
            if (
                self.window_resolution[0] != self.tiles[0] * self.tile_resolution[0]
                or self.window_resolution[1] != self.tiles[1] * self.tile_resolution[1]
            ):
                raise ValueError(
                    "window_resolution must equal tiles * tile_resolution. "
                    f"Got window_resolution={self.window_resolution}, tiles={self.tiles}, "
                    f"tile_resolution={self.tile_resolution}."
                )
        
        elif num_nones > 1:
            default_tiles = (1, 1)
            default_tile_resolution = (64, 64)
            if self.tiles is not None:
                self.tile_resolution = default_tile_resolution
                self.window_resolution = (64*self.tiles[0], 64*self.tiles[1])
            elif self.tile_resolution is not None:
                self.tiles = default_tiles
                self.window_resolution = (self.tile_resolution[0], self.tile_resolution[1])
            elif self.window_resolution is not None:
                self.tiles = default_tiles
                self.tile_resolution = (self.window_resolution[0], self.window_resolution[1])
            else:
                self.tiles = default_tiles
                self.tile_resolution = default_tile_resolution
                self.window_resolution = (default_tile_resolution[0]*default_tiles[0], default_tile_resolution[1]*default_tiles[1])
        else:
            if self.tiles is None:
                self.tiles = (
                    self.window_resolution[0] // self.tile_resolution[0],
                    self.window_resolution[1] // self.tile_resolution[1],
                )
            elif self.tile_resolution is None:
                self.tile_resolution = (
                    self.window_resolution[0] // self.tiles[0],
                    self.window_resolution[1] // self.tiles[1],
                )
            elif self.window_resolution is None:
                self.window_resolution = (
                    self.tiles[0] * self.tile_resolution[0],
                    self.tiles[1] * self.tile_resolution[1],
                )

        if self.num_scenes is None:
            self.num_scenes = self.tiles[0] * self.tiles[1]
        elif self.num_scenes > self.tiles[0] * self.tiles[1]:
            raise ValueError(f'{self.num_scenes=} can\'t fit into {self.tiles=}')

        # TODO: remove and merge with num_scenes 
        if self.batch_inner_dim is None:
            self.batch_inner_dim = self.tiles[0] * self.tiles[1]
        elif self.batch_inner_dim != self.tiles[0] * self.tiles[1]:
            raise ValueError("batch_inner_dim must equal tiles[0] * tiles[1]")

    def process_device(self) -> None:
        if self.device is not None and self.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device: {self.device}")
        # Placeholder to keep CPU for now; extend with torch checks if needed.
        self.device = 'cpu'

    def __post_init__(self) -> None:
        self.process_resolution()
        self.process_device()

    @classmethod
    def from_config(cls, cfg: P3DConfig | dict | None = None, **overrides) -> P3DConfig:
        if cfg is not None and overrides:
            raise ValueError("cfg and additional keyword arguments cannot be used together")


        if isinstance(cfg, cls):
            cfg_dict = asdict(cfg)
        elif isinstance(cfg, dict):
            cfg_dict = cfg
        else:
            cfg_dict = {}

        if overrides:
            cfg_dict.update(overrides)

        # cfg_diff = {k: v for k, v in cfg_dict.items() if k in default_cfg and v != default_cfg[k]}
        # print(cfg_diff)
        # print(f"P3DConfig non-defaults: {cfg_diff}")
        # print(f"Default {default_cfg}")

        # return cls(**cfg_diff)
        return cls(**cfg_dict)

    def build_prc(self) -> str:
        if self.window_resolution is None:
            # Ensure resolution is set (should be done in process_resolution already)
            self.process_resolution()
        prc = []
        prc.append(f"window-type {'offscreen' if self.offscreen else 'onscreen'}\n")
        prc.append(f"win-size {self.window_resolution[0]} {self.window_resolution[1]}\n")
        if self.panda3d_backend == 'arm':
            prc.append('gl-version 3 2\n')
        else:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            prc.append('threading-model Cull/Draw\n')
        prc.append(self.extra_prc_file_data)

        print(f"PRC: {prc}")
        return ''.join(prc)

    def __repr__(self):
        repr_string = f"P3DConfig(\n"
        for key, value in asdict(self).items():
            repr_string += f"    {key}: {value!r}\n"
        repr_string += ")"
        return repr_string

