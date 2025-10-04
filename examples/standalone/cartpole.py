import math
import numpy as np
from dataclasses import dataclass

from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import Texture, GeomEnums, Shader, OmniBoundingVolume, loadPrcFileData


@dataclass
class Config:
    # Window / context
    offscreen: bool = False
    gl_version: tuple[int, int] = (3, 2)
    tile_w: int = 320
    tile_h: int = 240

    # Scene sizes
    num_objects: int = 64
    num_cameras: int = 64
    grid_step: float = 5.0

    # Physics
    dt: float = 1 / 60
    force_mag: float = 10.0
    rand_interval: float = 0.15
    seed: int = 0

    # Visual scales
    cart_size: tuple[float, float, float] = (1.2, 0.8, 0.5)
    pole_size: tuple[float, float, float] = (0.1, 0.1, 2.0)

    # Camera follow (fixed angle or explicit offset)
    cam_left_angle_deg: float = 30.0
    cam_distance: float = 6.0
    cam_height: float = 1.8
    cam_offset: tuple[float, float, float] | None = None  # if set, overrides distance/angle/height

    # Projection
    fov_y_deg: float = 55.0
    z_near: float = 0.05
    z_far: float = 200.0

    # Visibility control
    hide_others: bool = True

    # Damping (to stabilise numerical integration)
    linear_damping: float = 0.5   # per-second damping for x_dot
    angular_damping: float = 0.2  # per-second damping for theta_dot


VERT_SRC = r"""#version 150
// Per-instance buffers (cart, pole, rail)
uniform samplerBuffer cart_matbuf;   // (N*4) texels: per-cart model matrix columns
uniform samplerBuffer pole_matbuf;   // (N*4) texels: per-pole model matrix columns
uniform samplerBuffer rail_matbuf;   // (N*4) texels: per-rail model matrix columns
uniform samplerBuffer colbuf;        // (N) texels: per-instance color (cart) — pole/rail use constant tint

// Per-view buffers
uniform samplerBuffer viewbuf;       // (K*4) texels: per-view VP matrix columns
uniform samplerBuffer tilebuf;       // (K)   texels: per-view tile rect (u0,u1,v0,v1)
uniform int K;                       // number of views

// Select which geometry this draw is for
uniform int geomType;                // 0 = cart, 1 = pole, 2 = rail

in vec4 p3d_Vertex;
in vec3 p3d_Normal;

out vec3 v_normal;
out vec4 v_color;
flat out int v_view;
flat out int v_visible;

void main() {
    int view = gl_InstanceID % K;
    int inst = gl_InstanceID / K;

    // Choose per-instance matrix from the appropriate buffer
    int mbase = inst * 4;
    vec4 c0, c1, c2, c3;
    if (geomType == 1) {
        c0 = texelFetch(pole_matbuf, mbase + 0);
        c1 = texelFetch(pole_matbuf, mbase + 1);
        c2 = texelFetch(pole_matbuf, mbase + 2);
        c3 = texelFetch(pole_matbuf, mbase + 3);
    } else if (geomType == 2) {
        c0 = texelFetch(rail_matbuf, mbase + 0);
        c1 = texelFetch(rail_matbuf, mbase + 1);
        c2 = texelFetch(rail_matbuf, mbase + 2);
        c3 = texelFetch(rail_matbuf, mbase + 3);
    } else {
        c0 = texelFetch(cart_matbuf, mbase + 0);
        c1 = texelFetch(cart_matbuf, mbase + 1);
        c2 = texelFetch(cart_matbuf, mbase + 2);
        c3 = texelFetch(cart_matbuf, mbase + 3);
    }
    mat4 M = mat4(c0, c1, c2, c3);

    int vbase = view * 4;
    vec4 vc0 = texelFetch(viewbuf, vbase + 0);
    vec4 vc1 = texelFetch(viewbuf, vbase + 1);
    vec4 vc2 = texelFetch(viewbuf, vbase + 2);
    vec4 vc3 = texelFetch(viewbuf, vbase + 3);
    mat4 VP = mat4(vc0, vc1, vc2, vc3);

    vec4 clip = VP * (M * p3d_Vertex);

    vec4 tile = texelFetch(tilebuf, view); // (u0,u1,v0,v1) in [0,1]
    float x0 = 2.0 * tile.x - 1.0;
    float x1 = 2.0 * tile.y - 1.0;
    float y0 = 2.0 * tile.z - 1.0;
    float y1 = 2.0 * tile.w - 1.0;
    vec2 scale_xy = vec2(0.5 * (x1 - x0), 0.5 * (y1 - y0));
    vec2 offset_xy = vec2(0.5 * (x0 + x1), 0.5 * (y0 + y1));

    gl_Position = vec4(clip.xy * scale_xy + offset_xy * clip.w, clip.z, clip.w);

    v_normal = normalize(mat3(M) * p3d_Normal);
    vec4 cart_color = texelFetch(colbuf, inst);
    if (geomType == 1) {
        v_color = vec4(1.0, 0.7, 0.2, 1.0);
    } else if (geomType == 2) {
        v_color = vec4(0.2, 0.2, 0.2, 1.0);
    } else {
        v_color = cart_color;
    }
    v_view = view;
    v_visible = (inst == view) ? 1 : 0;
}
"""


FRAG_SRC = r"""#version 150
in vec3 v_normal;
in vec4 v_color;
flat in int v_view;
flat in int v_visible;
out vec4 fragColor;
uniform samplerBuffer tilebuf;   // per-view tile rect (u0,u1,v0,v1)
uniform vec2 screenSize;         // (width, height)
uniform int hideOthers;          // 1 = show only followed, 0 = show all
void main() {
    // Clip to the current view's tile region (no loops)
    vec4 tile = texelFetch(tilebuf, v_view);
    float x0 = tile.x * screenSize.x;
    float x1 = tile.y * screenSize.x;
    float y0 = tile.z * screenSize.y;
    float y1 = tile.w * screenSize.y;
    if (gl_FragCoord.x < x0 || gl_FragCoord.x >= x1 || gl_FragCoord.y < y0 || gl_FragCoord.y >= y1) {
        discard;
    }
    // Optionally render only the instance that matches this view
    if (hideOthers == 1 && v_visible == 0) discard;
    vec3 n = normalize(v_normal);
    float ndl = max(dot(n, normalize(vec3(0.4, 0.6, 0.7))), 0.0);
    vec3 col = v_color.rgb * (0.2 + 0.8 * ndl);
    fragColor = vec4(col, v_color.a);
}
"""


def setup_buffer_texture(name: str, texels: int, usage=GeomEnums.UH_dynamic) -> Texture:
    tex = Texture(name)
    tex.setupBufferTexture(texels, Texture.T_float, Texture.F_rgba32, usage)
    tex.set_keep_ram_image(True)
    return tex


def pack_columns(mat_batch: np.ndarray) -> np.ndarray:
    # (B,4,4) -> (B,4,4) with columns ready for column fetch
    return mat_batch.transpose(0, 2, 1).astype(np.float32, copy=False)


class CartPoleMultiView(ShowBase):
    def __init__(self, cfg: Config):
        # Arrange a view grid from config
        N = int(cfg.num_objects)
        K = int(cfg.num_cameras)
        cols = int(math.ceil(math.sqrt(K)))
        rows = int(math.ceil(K / cols))
        win_w, win_h = cols * cfg.tile_w, rows * cfg.tile_h
        loadPrcFileData('', f'show-frame-rate-meter 1\n')
        loadPrcFileData('', f'sync-video 0\n')
        loadPrcFileData('', f'gl-version {cfg.gl_version[0]} {cfg.gl_version[1]}\n')
        loadPrcFileData('', f'win-size {win_w} {win_h}\n')
        if cfg.offscreen:
            loadPrcFileData('', 'window-type offscreen\n')
        else:
            loadPrcFileData('', 'window-type onscreen\n')

        super().__init__()

        self.cfg = cfg
        self.N = N
        self.K = K
        self.cols = cols
        self.rows = rows
        self.step = float(cfg.grid_step)
        self.dt = float(cfg.dt)

        # Visual sizes
        self.cart_size = cfg.cart_size
        self.pole_size = cfg.pole_size

        # Physics params
        self.g = 9.8
        self.m_c = 1.0
        self.m_p = 0.1
        self.total_m = self.m_c + self.m_p
        self.length = 0.5
        self.poleml = self.m_p * self.length
        self.force_mag = float(cfg.force_mag)

        # State (N,1)
        self.x         = np.zeros((self.N,1), np.float32)
        self.x_dot     = np.zeros((self.N,1), np.float32)
        self.theta     = np.random.uniform(-0.03, 0.03, size=(self.N,1)).astype(np.float32)
        self.theta_dot = np.zeros((self.N,1), np.float32)
        self.force     = np.zeros((self.N,1), np.float32)

        # Random control for carts 1..N-1
        self.rng = np.random.default_rng(cfg.seed)
        self.rand_interval = float(cfg.rand_interval)
        self._rand_accum = 0.0

        # Keyboard control (cart 0)
        self._key_left = False
        self._key_right = False
        self.accept('arrow_left',  self._on_key, ['left', True])
        self.accept('arrow_left-up',  self._on_key, ['left', False])
        self.accept('arrow_right', self._on_key, ['right', True])
        self.accept('arrow_right-up', self._on_key, ['right', False])

        # Camera params (fixed-angle follow)
        self.cam_left_angle_deg = float(cfg.cam_left_angle_deg)
        self.cam_distance = float(cfg.cam_distance)
        self.cam_height = float(cfg.cam_height)
        self.cam_offset = cfg.cam_offset

        # View layout and base positions
        self.base_xy = self._make_grid_offsets(self.N, self.step)

        # Build scene (one box model reused for cart and pole in two draws)
        self._build_model()
        self._build_buffers()
        self._prime_buffers()

        self.taskMgr.add(self._update, "update")
        # FPS print control
        self._fps_print_interval = 1.0
        self._fps_last_print_time = 0.0

    def _make_grid_offsets(self, N: int, step: float) -> np.ndarray:
        cols = int(math.ceil(math.sqrt(N)))
        rows = int(math.ceil(N / cols))
        xs, ys = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
        x = xs.ravel()[:N] * step
        y = ys.ravel()[:N] * step
        x -= ((cols - 1) * step) * 0.5
        y -= ((rows - 1) * step) * 0.5
        return np.stack([x, y], axis=1).astype(np.float32)

    def _build_model(self):
        shader = Shader.make(Shader.SL_GLSL, VERT_SRC, FRAG_SRC)

        # Cart model pass (geomType = 0)
        self.cart_model = self.loader.loadModel("models/box")
        self.cart_model.clearModelNodes(); self.cart_model.flattenStrong()
        self.cart_model.reparentTo(self.render)
        self.cart_model.setColor(1.0, 1.0, 1.0, 1.0)
        self.cart_model.setScale(*self.cart_size)
        self.cart_model.set_instance_count(self.N * self.K)
        self.cart_model.setShader(shader)
        self.cart_model.node().setBounds(OmniBoundingVolume())
        self.cart_model.node().setFinal(True)
        self.cart_model.setShaderInput("geomType", 0)

        # Pole model pass (geomType = 1)
        self.pole_model = self.loader.loadModel("models/box")
        self.pole_model.clearModelNodes(); self.pole_model.flattenStrong()
        self.pole_model.reparentTo(self.render)
        self.pole_model.setColor(1.0, 1.0, 1.0, 1.0)
        self.pole_model.setScale(*self.pole_size)
        self.pole_model.set_instance_count(self.N * self.K)
        self.pole_model.setShader(shader)
        self.pole_model.node().setBounds(OmniBoundingVolume())
        self.pole_model.node().setFinal(True)
        self.pole_model.setShaderInput("geomType", 1)

        # Rail model pass (geomType = 2)
        self.rail_model = self.loader.loadModel("models/box")
        self.rail_model.clearModelNodes(); self.rail_model.flattenStrong()
        self.rail_model.reparentTo(self.render)
        self.rail_model.setColor(1.0, 1.0, 1.0, 1.0)
        # base model scale does not matter; we scale via instance matrices
        self.rail_model.set_instance_count(self.N * self.K)
        self.rail_model.setShader(shader)
        self.rail_model.node().setBounds(OmniBoundingVolume())
        self.rail_model.node().setFinal(True)
        self.rail_model.setShaderInput("geomType", 2)

    def _build_buffers(self):
        # Per-instance transforms
        self.cart_matbuf = setup_buffer_texture("cart_matbuf", self.N * 4, GeomEnums.UH_dynamic)
        self.pole_matbuf = setup_buffer_texture("pole_matbuf", self.N * 4, GeomEnums.UH_dynamic)
        self.colbuf      = setup_buffer_texture("colbuf",      self.N,     GeomEnums.UH_dynamic)
        self.rail_matbuf = setup_buffer_texture("rail_matbuf", self.N * 4, GeomEnums.UH_dynamic)

        # Per-view
        self.viewbuf = setup_buffer_texture("viewbuf", self.K * 4, GeomEnums.UH_dynamic)
        self.tilebuf = setup_buffer_texture("tilebuf", self.K,     GeomEnums.UH_dynamic)

        # Bind common shader inputs
        for np_model in (self.cart_model, self.pole_model, self.rail_model):
            np_model.setShaderInput("cart_matbuf", self.cart_matbuf)
            np_model.setShaderInput("pole_matbuf", self.pole_matbuf)
            np_model.setShaderInput("rail_matbuf", self.rail_matbuf)
            np_model.setShaderInput("colbuf", self.colbuf)
            np_model.setShaderInput("viewbuf", self.viewbuf)
            np_model.setShaderInput("tilebuf", self.tilebuf)
            np_model.setShaderInput("K", self.K)
            np_model.setShaderInput("screenSize", (float(self.win.getXSize()), float(self.win.getYSize())))
            np_model.setShaderInput("hideOthers", 1 if self.cfg.hide_others else 0)

    def _prime_buffers(self):
        # Identity transforms initially
        I = np.zeros((self.N, 4, 4), np.float32)
        I[:, 0, 0] = 1; I[:, 1, 1] = 1; I[:, 2, 2] = 1; I[:, 3, 3] = 1
        self.cart_matbuf.set_ram_image(pack_columns(I).tobytes(order='C'))
        self.pole_matbuf.set_ram_image(pack_columns(I).tobytes(order='C'))
        self.rail_matbuf.set_ram_image(pack_columns(I).tobytes(order='C'))

        # Colors (per cart)
        col = np.zeros((self.N, 4), np.float32)
        col[:, 0] = np.linspace(0.1, 1.0, self.N)
        col[:, 1] = 0.6
        col[:, 2] = 1.0
        col[:, 3] = 1.0
        self.colbuf.set_ram_image(col.tobytes(order='C'))

        # Tiles
        tiles = np.zeros((self.K, 4), np.float32)
        for v in range(self.K):
            cx = v % self.cols
            cy = v // self.cols
            u0 = cx / self.cols
            u1 = (cx + 1) / self.cols
            v0 = 1.0 - (cy + 1) / self.rows
            v1 = 1.0 - cy / self.rows
            tiles[v] = (u0, u1, v0, v1)
        self.tilebuf.set_ram_image(tiles.tobytes(order='C'))

        # Init viewbuf
        VP = np.zeros((self.K, 4, 4), np.float32)
        VP[:, 0, 0] = 1; VP[:, 1, 1] = 1; VP[:, 2, 2] = 1; VP[:, 3, 3] = 1
        self.viewbuf.set_ram_image(pack_columns(VP).tobytes(order='C'))

    def _perspective_batch(self, K: int, fov_y_deg: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
        P = np.zeros((K, 4, 4), dtype=np.float32)
        P[:, 0, 0] = f / aspect
        P[:, 1, 1] = f
        P[:, 2, 2] = (z_far + z_near) / (z_near - z_far)
        P[:, 2, 3] = (2 * z_far * z_near) / (z_near - z_far)
        P[:, 3, 2] = -1.0
        return P

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n = np.maximum(n, 1e-8)
        return v / n

    def _look_at_batch(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        f = self._normalize(target - eye)
        s = self._normalize(np.cross(f, up))
        u = np.cross(s, f)
        V = np.zeros((eye.shape[0], 4, 4), dtype=np.float32)
        V[:, 0, 0:3] = s
        V[:, 1, 0:3] = u
        V[:, 2, 0:3] = -f
        V[:, 3, 3] = 1.0
        V[:, 0, 3] = -np.sum(s * eye, axis=1)
        V[:, 1, 3] = -np.sum(u * eye, axis=1)
        V[:, 2, 3] =  np.sum(f * eye, axis=1)
        return V

    def _step_physics(self, dt: float):
        x         = self.x
        x_dot     = self.x_dot
        theta     = self.theta
        theta_dot = self.theta_dot
        force     = self.force

        costheta = np.cos(theta, dtype=np.float32)
        sintheta = np.sin(theta, dtype=np.float32)

        temp = (force + self.poleml * (theta_dot * theta_dot) * sintheta) / self.total_m
        thetaacc = (self.g * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.m_p * (costheta * costheta) / self.total_m)
        )
        xacc = temp - (self.poleml * thetaacc * costheta) / self.total_m

        # Semi-implicit Euler with exponential damping
        x_dot     = x_dot     + dt * xacc
        theta_dot = theta_dot + dt * thetaacc
        if self.cfg.linear_damping > 0.0 or self.cfg.angular_damping > 0.0:
            lin_damp = math.exp(-self.cfg.linear_damping * dt)
            ang_damp = math.exp(-self.cfg.angular_damping * dt)
            x_dot     = x_dot     * lin_damp
            theta_dot = theta_dot * ang_damp
        x     = x     + dt * x_dot
        theta = theta + dt * theta_dot

        self.x = x
        self.x_dot = x_dot
        self.theta = theta
        self.theta_dot = theta_dot

    def _compute_transforms(self, t: float):
        # Cart world pos = [base_x + x, base_y, 0]
        base_x = self.base_xy[:, 0:1]
        base_y = self.base_xy[:, 1:2]
        cart_pos = np.concatenate([base_x + self.x, base_y, np.zeros_like(self.x)], axis=1).astype(np.float32)

        # Cart transform matrices
        M_cart = np.zeros((self.N, 4, 4), np.float32)
        M_cart[:, 0, 0] = 1; M_cart[:, 1, 1] = 1; M_cart[:, 2, 2] = 1; M_cart[:, 3, 3] = 1
        M_cart[:, 0:3, 3] = cart_pos

        # Apply cart scale in model space: M_cart = T_cart @ S_cart
        S_cart = np.zeros((self.N, 4, 4), np.float32)
        S_cart[:, 0, 0] = self.cart_size[0]
        S_cart[:, 1, 1] = self.cart_size[1]
        S_cart[:, 2, 2] = self.cart_size[2]
        S_cart[:, 3, 3] = 1.0
        M_cart = M_cart @ S_cart

        # Pole = Cart_T * T_hinge * R_y(theta) * T_base_up * S_pole
        # Hinge at cart top center; rotate in XZ plane (around Y); base at hinge
        z_off = (self.cart_size[2] * 0.5)
        T_hinge = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,z_off],[0,0,0,1]], dtype=np.float32)
        T_hinge = np.broadcast_to(T_hinge, (self.N,4,4))

        th = self.theta.reshape(-1,1).astype(np.float32, copy=False)
        c, s = np.cos(th), np.sin(th)
        R_y = np.zeros((self.N, 4, 4), np.float32)
        R_y[:, 0, 0] =  c[:, 0]; R_y[:, 0, 2] = s[:, 0]
        R_y[:, 1, 1] = 1.0
        R_y[:, 2, 0] = -s[:, 0]; R_y[:, 2, 2] = c[:, 0]
        R_y[:, 3, 3] = 1.0

        # Apply pole scale in model space and lift center so base lies at hinge
        T_base_up = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,self.pole_size[2]*0.5],[0,0,0,1]], dtype=np.float32)
        T_base_up = np.broadcast_to(T_base_up, (self.N,4,4))
        S_pole = np.zeros((self.N, 4, 4), np.float32)
        S_pole[:, 0, 0] = self.pole_size[0]
        S_pole[:, 1, 1] = self.pole_size[1]
        S_pole[:, 2, 2] = self.pole_size[2]
        S_pole[:, 3, 3] = 1.0

        local_pole = R_y @ (T_base_up @ S_pole)
        M_pole = M_cart @ (T_hinge @ local_pole)
        
        # Rail: long thin box centered at (base_x, base_y, 0), fixed in world
        rail_len = 6.0
        M_rail = np.zeros((self.N, 4, 4), np.float32)
        M_rail[:, 0, 0] = 1; M_rail[:, 1, 1] = 1; M_rail[:, 2, 2] = 1; M_rail[:, 3, 3] = 1
        M_rail[:, 0:3, 3] = np.c_[base_x[:, 0], base_y[:, 0], np.zeros(self.N, np.float32)]
        S_rail = np.zeros((self.N, 4, 4), np.float32)
        S_rail[:, 0, 0] = rail_len
        S_rail[:, 1, 1] = 0.05
        S_rail[:, 2, 2] = 0.05
        S_rail[:, 3, 3] = 1.0
        M_rail = M_rail @ S_rail

        return M_cart, M_pole, M_rail

    def _compute_viewproj(self, t: float) -> np.ndarray:
        # One view per env i, camera follows cart i with fixed left angle
        idx = np.arange(self.K, dtype=np.int32)
        base_x = self.base_xy[:, 0]
        base_y = self.base_xy[:, 1]
        target = np.c_[base_x[idx] + self.x[idx,0], base_y[idx], np.zeros(self.K, np.float32)].astype(np.float32)
        ang = math.radians(self.cam_left_angle_deg)
        offset_x = -math.sin(ang) * self.cam_distance
        offset_y = -math.cos(ang) * self.cam_distance
        eye = np.zeros_like(target)
        eye[:, 0] = target[:, 0] + offset_x
        eye[:, 1] = target[:, 1] + offset_y
        eye[:, 2] = target[:, 2] + self.cam_height

        up = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (self.K, 1))
        V = self._look_at_batch(eye.astype(np.float32), target, up)
        P = self._perspective_batch(self.K, fov_y_deg=self.cfg.fov_y_deg, aspect=1.3333333, z_near=self.cfg.z_near, z_far=self.cfg.z_far)
        return (P @ V).astype(np.float32)

    def _update(self, task):
        # Fixed step physics
        dt = max(1e-3, min(1.0/30.0, globalClock.getDt()))
        steps = max(1, int(round(dt / self.dt)))
        sub_dt = dt / steps
        for _ in range(steps):
            # Update controls: keyboard for cart 0, random for others
            f0 = 0.0
            if self._key_left and not self._key_right:
                f0 = -self.force_mag
            elif self._key_right and not self._key_left:
                f0 = self.force_mag
            self.force[0, 0] = float(f0)

            self._rand_accum += sub_dt
            if self._rand_accum >= self.rand_interval:
                self._rand_accum -= self.rand_interval
                if self.N > 1:
                    self.force[1:, 0] = self.rng.uniform(-self.force_mag, self.force_mag, size=(self.N - 1,))

            self._step_physics(sub_dt)

        # Upload transforms (column-packed)
        t = globalClock.getFrameTime()
        M_cart, M_pole, M_rail = self._compute_transforms(t)
        self.cart_matbuf.set_ram_image(pack_columns(M_cart).tobytes(order='C'))
        self.pole_matbuf.set_ram_image(pack_columns(M_pole).tobytes(order='C'))
        self.rail_matbuf.set_ram_image(pack_columns(M_rail).tobytes(order='C'))

        # Upload per-view VP matrices
        VP = self._compute_viewproj(t)
        self.viewbuf.set_ram_image(pack_columns(VP).tobytes(order='C'))

        # Print FPS once per second (average frame rate)
        now = globalClock.getRealTime()
        if now - self._fps_last_print_time >= self._fps_print_interval:
            fps = globalClock.getAverageFrameRate()
            try:
                print(f"FPS: {fps:.1f}")
            except Exception:
                print("FPS:", float(fps))
            self._fps_last_print_time = now

        return task.cont

    # Keyboard handler
    def _on_key(self, which: str, down: bool):
        if which == 'left':
            self._key_left = down
        elif which == 'right':
            self._key_right = down

    def draw(self):
        pass


if __name__ == "__main__":
    cfg = Config(
        offscreen=False,
        gl_version=(3, 2),
        tile_w=84, tile_h=64,
        num_objects=64, num_cameras=64,
        # offscreen gives: 1024 - 36, 512 - 140, 256 - 540, 128 - 2k, 64 - 3.8k, 32 - 4.2k
        grid_step=5.0,
        dt=1/120,
        force_mag=10.0,
        rand_interval=0.15,
        seed=0,
        cart_size=(1.2, 0.8, 0.5),
        pole_size=(0.1, 0.1, 2.0),
        cam_left_angle_deg=30.0,
        cam_distance=6.0,
        cam_height=1.8,
        cam_offset=None,
        fov_y_deg=55.0,
        z_near=1,
        z_far=50.0,
        hide_others=True,
    )
    app = CartPoleMultiView(cfg)
    app.run()


