from typing import Literal

from direct.showbase.ShowBase import ShowBase

from .shader_context import PBRShaderContext


class PBRLight(PBRShaderContext):
    """Global lighting controller (ambient + directional) broadcasting to all registered nodes."""
    def __init__(self, showbase: ShowBase,
                 ambient: tuple[float, float, float] = (0.2, 0.2, 0.25),
                 dir_dir: tuple[float, float, float] = (0.4, -0.6, -0.7),
                 dir_col: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 strength: float = 1.0,
                 backend: Literal[ "loop", "instanced"] = "instanced"
                 ) -> None:
        super().__init__(showbase, backend=backend)

        # Broadcast initial values
        self.set_strength(strength)
        self.set_directional(dir_dir, dir_col)
        self.set_ambient(ambient)
        # Register this light globally
        self._register_self()

    def set_strength(self, strength: float) -> None:
        self.strength = float(strength)
        self._set_shader_input('lightingStrength', self.strength)

    def set_directional(self, dir_dir: tuple[float, float, float], dir_col: tuple[float, float, float]) -> None:
        self.dir_dir = tuple(float(x) for x in dir_dir)
        self.dir_col = tuple(float(x) for x in dir_col)
        self._set_shader_input('dirLightDir', self.dir_dir)
        self._set_shader_input('dirLightCol', self.dir_col)

    def set_ambient(self, amb_col: tuple[float, float, float]) -> None:
        self.ambient = tuple(float(x) for x in amb_col)
        self._set_shader_input('ambientCol', self.ambient)

    def _register_self(self) -> None:
        # Ensure light singleton on ShowBase
        self.base._pbr_light = self


