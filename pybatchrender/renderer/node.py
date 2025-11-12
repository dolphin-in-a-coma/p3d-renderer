import torch

from collections.abc import Sequence
from typing import Literal

from panda3d.core import Texture, OmniBoundingVolume, NodePath
from direct.showbase.ShowBase import ShowBase

from .shader_context import PBRShaderContext


class PBRNode(PBRShaderContext):
    def __init__(self,
                showbase,
                model_path: str | None, # NOTE: setting None is not fully functional yet
                num_scenes: int = 1,
                instances_per_scene: int = 1,
                texture: Texture | str | bool | None = None,
                model_pivot_relative_point: tuple[float, float, float] | None = None,
                model_scale: float | Sequence[float] | None = None,
                model_scale_units: Literal["relative", "absolute"] = "relative",
                positions: torch.Tensor | None = None,
                hprs: torch.Tensor | None = None,
                scales: torch.Tensor | None = None,
                colors: torch.Tensor | None = None,
                backend: Literal[ "loop", "instanced"] = "instanced",
                shared_across_scenes: bool = False,
                parent: 'PBRNode | NodePath | None' = None, # NOTE: parent is not fully functional yet
                name: str | None = None
                ) -> None:

        if shared_across_scenes:
            pass

        if positions is not None or \
            colors is not None or \
            scales is not None or \
            hprs is not None:
            print("WARNING: initializing positions, colors, scales are not implemented yet, they will be ignored")
        super().__init__(showbase, backend=backend)
        self.num_scenes = int(num_scenes)
        self.instances_per_scene = int(instances_per_scene)
        self.shared_across = bool(shared_across_scenes)
        self.total_instances = self.num_scenes * self.instances_per_scene # if not self.shared_across else self.instances_per_scene
        self.buf_instances = self.instances_per_scene if self.shared_across else self.total_instances
        self.model_pivot_relative_point = model_pivot_relative_point
        self.model_scale = model_scale
        self.model_scale_units = model_scale_units
        
        # TODO: remove for now
        # parent_np = None
        # if isinstance(parent, PBRNode):
        #     parent_np = parent.np
        # elif isinstance(parent, NodePath):
        #     parent_np = parent
        # else:
        #     parent_np = self.base.render
        
        if model_path:
            self.np = self.base.loader.loadModel(model_path)
            self._apply_model_scale()
            self.np.clearModelNodes(); self.np.flattenStrong()
            self.np.reparentTo(self.base.render) # NOTE: all nodes are children of the base render
            self.pivot_to_rel(self.model_pivot_relative_point)
            self.np.set_instance_count(self.total_instances)
            self.np.setShader(self._make_shader()) # TODO: implement CUDA way too with CUDA specific shaders
            self.np.node().setBounds(OmniBoundingVolume())
            self.np.node().setFinal(True)
            self.has_geometry = True
        else:
            # Create an empty transform node, NOTE: empty nodes are not fully supported yet
            self.np = self.base.render.attachNewNode(name or 'pbr_empty')
            self.has_geometry = False
            # TODO: implement geometry that affects children nodes
        
        # Register this nodepath on the ShowBase registry
        self._register_self()
        self._attempt_camera_connect()
        self._attempt_light_connect()
        
        # allocate buffers only if there is geometry to render
        if model_path:
            self.matbuf = self._setup_buffer_texture('pbr_matbuf', self.buf_instances*4)
            self.colbuf = self._setup_buffer_texture('pbr_colbuf', self.buf_instances)
            self._set_shader_input('matbuf', self.matbuf)
            self._set_shader_input('colbuf', self.colbuf)
            self._set_shader_input('instancesPerScene', max(1, self.buf_instances//max(1,self.num_scenes)))
            self._set_shader_input('shareAcrossScenes', 1 if self.shared_across else 0)
 
        # defaults (safe to broadcast even for empty nodes)
        self._set_lighting_strength(0.0)
        self._set_lighting(dir_dir=(0.4,0.6,0.7), dir_col=(1,1,1), amb_col=(0.2,0.2,0.25))
        self.set_texture(texture)

        # init identity mats and white color
        # TODO: figure out why this is needed, and edit to use positions, scales, hprs, colors, that are in the arguments
        if model_path:
            self.transforms_b44 = torch.zeros((self.buf_instances, 4, 4), dtype=torch.float32)
            self.transforms_b44[:,0,0]=1; self.transforms_b44[:,1,1]=1; self.transforms_b44[:,2,2]=1; self.transforms_b44[:,3,3]=1
            self.rot3_b33 = torch.zeros((self.buf_instances, 3, 3), dtype=torch.float32)
            self.rot3_b33[:,0,0]=1; self.rot3_b33[:,1,1]=1; self.rot3_b33[:,2,2]=1
            self.scale_b11 = torch.ones((self.buf_instances, 1, 1), dtype=torch.float32)
            self._upload_current_transforms()
            col = torch.ones((self.buf_instances, 4), dtype=torch.float32)
            self.colbuf.set_ram_image(col.contiguous().cpu().numpy().tobytes(order='C'))

    def _upload_mat(self, mats: torch.Tensor) -> None:
        if not getattr(self, 'has_geometry', True):
            return
        mats = type(self)._pack_columns(mats)
        self.matbuf.set_ram_image(mats.contiguous().cpu().numpy().tobytes(order='C'))

    def _upload_current_transforms(self) -> None:
        if not getattr(self, 'has_geometry', True):
            return
        # B = self.buf_instances
        # R = self.rot3_b33.reshape(int(B), 3, 3)
        # s = self.scale_b1.reshape(int(B), 1, 1)
        # print(self.rot3_b33.shape, self.scale_b1.shape)
        # print((self.scale_b1[:,None]).shape)
        self.transforms_b44[:,0:3,0:3] = self.rot3_b33 * self.scale_b11
        # print(self.transforms_b44[:,0:3,0:3])
        self._upload_mat(self.transforms_b44)

    def set_positions(self, pos_si3: torch.Tensor, lazy: bool = False) -> None:
        if not getattr(self, 'has_geometry', True):
            return
        pos = torch.as_tensor(pos_si3, dtype=torch.float32).reshape(-1, 3)
        self.transforms_b44[:,0:3,3]=pos[:self.buf_instances]
        if not lazy:
            self._upload_current_transforms()

    def set_hprs(self, hpr_si3: torch.Tensor, lazy: bool = False) -> None:
        if not getattr(self, 'has_geometry', True):
            return
        hpr = torch.as_tensor(hpr_si3, dtype=torch.float32).reshape(-1, 3)
        R3 = type(self)._rotation_mats_from_hpr(hpr[:self.buf_instances])
        self.rot3_b33[:,:,:] = R3
        if not lazy:
            self._upload_current_transforms()

    def set_scales(self, scale_si1: torch.Tensor | float, lazy: bool = False) -> None:
        if not getattr(self, 'has_geometry', True):
            return
        if isinstance(scale_si1, (float, int)):
            s = torch.full((self.buf_instances, 1, 1), float(scale_si1), dtype=torch.float32)
        else:
            s = torch.as_tensor(scale_si1, dtype=torch.float32).reshape(-1, 1, 1)
        self.scale_b11 = s[:self.buf_instances]
        if not lazy:
            self._upload_current_transforms()

    def set_colors(self, col_si4: torch.Tensor) -> None:
        if not getattr(self, 'has_geometry', True):
            return
        col = torch.as_tensor(col_si4, dtype=torch.float32).reshape(-1, 4)
        col = col[:self.buf_instances]
        self.colbuf.set_ram_image(col.contiguous().cpu().numpy().tobytes(order='C'))

    def set_transforms(self, mats_b44: torch.Tensor) -> None:
        """Upload full per-instance transform matrices (B,4,4)."""
        if not getattr(self, 'has_geometry', True):
            return
        mats_b44 = torch.as_tensor(mats_b44, dtype=torch.float32)
        self.transforms_b44 = mats_b44.clone()
        self.rot3_b33 = self.transforms_b44[:,0:3,0:3].clone()
        # assume uniform scale from 3x3 average of column norms
        col0 = torch.linalg.norm(self.rot3_b33[:,0,:], dim=1)
        col1 = torch.linalg.norm(self.rot3_b33[:,1,:], dim=1)
        col2 = torch.linalg.norm(self.rot3_b33[:,2,:], dim=1)
        s = (col0 + col1 + col2) / 3.0
        s = torch.clamp(s, min=1e-8)
        self.scale_b11 = s.reshape(-1, 1, 1).to(torch.float32)
        self.rot3_b33 = self.rot3_b33 / self.scale_b11
        self._upload_current_transforms()

    def _coerce_model_scale(self) -> tuple[tuple[float, float, float], bool] | None:
        """Return coerced scale tuple and whether it was originally uniform."""
        scale = self.model_scale
        if scale is None:
            return None

        if isinstance(scale, (int, float)):
            value = float(scale)
            return ((value, value, value), True)

        if isinstance(scale, Sequence):
            seq = tuple(float(v) for v in scale)
            if len(seq) == 0:
                raise ValueError("model_scale sequence cannot be empty")
            if len(seq) == 1:
                value = seq[0]
                return ((value, value, value), True)
            if len(seq) != 3:
                raise ValueError("model_scale sequence must have length 1 or 3")
            return (seq, False)

        raise TypeError("model_scale must be a number, a sequence of length 1 or 3, or None")

    def _apply_model_scale(self) -> None:
        if not getattr(self, 'has_geometry', True):
            # Nothing to scale
            return

        coerced = self._coerce_model_scale()
        if coerced is None:
            return

        scale_values, is_uniform = coerced
        if self.model_scale_units == "relative":
            self.np.setScale(*scale_values)
            return

        if self.model_scale_units != "absolute":
            raise ValueError(f"Unknown model_scale_units '{self.model_scale_units}'")

        bounds = self.np.getTightBounds()
        if not bounds:
            # Fall back to relative scaling if bounds are unavailable
            self.np.setScale(*scale_values)
            return

        min_pt, max_pt = bounds
        if min_pt is None or max_pt is None:
            self.np.setScale(*scale_values)
            return
        dims = (
            float(max_pt[0] - min_pt[0]),
            float(max_pt[1] - min_pt[1]),
            float(max_pt[2] - min_pt[2]),
        )
        longest = max(max(dims), 1e-8)

        if is_uniform:
            target = scale_values[0]
            factor = target / longest
            self.np.setScale(factor, factor, factor)
            return

        scale_components: list[float] = []
        for target_dim, source_dim in zip(scale_values, dims):
            denom = source_dim if abs(source_dim) >= 1e-8 else longest
            if abs(denom) < 1e-8:
                denom = 1.0
            scale_components.append(target_dim / denom)
        self.np.setScale(*scale_components)

    def _register_self(self) -> None:
        # Ensure registry exists and add this node once
        if not hasattr(self.base, '_pbr_nodes'):
            self.base._pbr_nodes = []
        if self not in self.base._pbr_nodes:
            self.base._pbr_nodes.append(self)

    def set_texture(self, texture: Texture | str | bool | None = None
    ) -> None:
        if texture is not None:
            if isinstance(texture, str):
                texture = self.base.loader.loadTexture(texture)
                self.np.setTexture(texture)
            elif isinstance(texture, Texture):
                self.np.setTexture(texture)
            self._set_shader_input('useTexture', 1.0) # NOTE: if True, default texture is attempted
        else:
            self._set_shader_input('useTexture', 0.0)

    def reparent_to(self, parent: 'PBRNode | NodePath') -> None:
        raise NotImplementedError("PBRNode.reparent_to is not implemented yet")
        # Implement by linking the geometry of this node to the geometry of the parent node
        # target = parent.np if isinstance(parent, PBRNode) else parent
        # self.np.reparentTo(target)

    def _set_lighting_strength(self, strength: float, overwrite: bool = False) -> None:
        if overwrite or not hasattr(self.base, '_pbr_light'):
            self._set_shader_input('lightingStrength', float(strength))

    def _set_lighting(self, dir_dir, dir_col, amb_col, overwrite: bool = False) -> None:
        if overwrite or not hasattr(self.base, '_pbr_light'):
            self._set_shader_input('dirLightDir', tuple(float(x) for x in dir_dir))
            self._set_shader_input('dirLightCol', tuple(float(x) for x in dir_col))
            self._set_shader_input('ambientCol',  tuple(float(x) for x in amb_col))

    def _attempt_camera_connect(self):
        if hasattr(self.base, '_pbr_cam') and self.base._pbr_cam:
            self.base._pbr_cam.attach(self)

    def _attempt_light_connect(self):
        if hasattr(self.base, '_pbr_light') and self.base._pbr_light:
            self.base._pbr_light.attach(self)

    def pivot_to_rel(self, relative_point: tuple[float, float, float] | None = None):
        """
        Recenter 'self.np' so the bounds-relative point 'rel' becomes its origin.
        rel=(0,0,0) -> min corner, (1,1,1) -> max corner; values outside [0,1] allowed.
        Modifies geometry in-place; no extra nodes remain.
        """
        if relative_point is None:
            return

        from panda3d.core import LPoint3
        model = self.np
        parent = model.getParent()
        tmp = parent.attachNewNode('tmp-pivot')
        model.wrtReparentTo(tmp)

        a, b = model.getTightBounds(tmp)
        if (not a) or (not b):
            model.wrtReparentTo(parent); tmp.removeNode(); return

        p = LPoint3(
            a.x + (b.x - a.x) * float(relative_point[0]),
            a.y + (b.y - a.y) * float(relative_point[1]),
            a.z + (b.z - a.z) * float(relative_point[2]),
        )
        model.setPos(tmp, -p)      # shift geometry relative to tmp so p -> (0,0,0)

        tmp.flattenStrong()         # bake child transforms into vertices
        model.wrtReparentTo(parent)
        tmp.removeNode()


