import math
from dataclasses import dataclass
import random
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import loadPrcFileData, LPoint3, AmbientLight, DirectionalLight, Vec3


# Utility: recenter model so bounds-relative point 'rel' becomes origin
def bake_pivot_to_rel(model, rel=(0.5, 0.5, 0.5)):
    parent = model.getParent()
    tmp = parent.attachNewNode('tmp-pivot')
    model.wrtReparentTo(tmp)

    a, b = model.getTightBounds(tmp)
    if not a or not b:
        model.wrtReparentTo(parent)
        tmp.removeNode()
        return

    p = LPoint3(
        a.x + (b.x - a.x) * rel[0],
        a.y + (b.y - a.y) * rel[1],
        a.z + (b.z - a.z) * rel[2],
    )
    model.setPos(tmp, -p)
    tmp.flattenStrong()
    model.wrtReparentTo(parent)
    tmp.removeNode()


@dataclass
class CartPoleConfig:
    # Window / context
    width: int = 640 
    height: int = 640
    show_fps: bool = True

    # Physics
    dt: float = 1 / 120
    g: float = 9.8
    m_c: float = 1.0
    m_p: float = 0.1
    force_mag: float = 10.0
    linear_damping: float = 0.5
    angular_damping: float = 0.2
    theta_init_range_deg: tuple[float, float] = (-5, 5)

    # Visual scales
    cart_size: tuple[float, float, float] = (1.2, 0.8, 0.5)
    pole_size: tuple[float, float, float] = (0.1, 0.1, 2.0)
    rail_size: tuple[float, float, float] = (6.0, 0.05, 0.05)

    cart_color: tuple[float, float, float, float] = (0.6, 0.8, 1.0, 1.0)
    pole_color: tuple[float, float, float, float] = (1.0, 0.7, 0.2, 1.0)
    rail_color: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)

    # Camera follow
    cam_position: tuple[float, float, float] = (-7, -12.0, 2.0)
    cam_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Reset thresholds
    theta_threshold_deg: float = 90.0  # pole fall threshold (|theta| > this)


class CarPoleLogic:
    def __init__(self, cfg: CartPoleConfig):
        self.cfg = cfg
        self.total_m = cfg.m_c + cfg.m_p
        self.pole_length = cfg.pole_size[2]
        self.poleml = cfg.m_p * self.pole_length # TODO: add actual physics, given that pole is pointed not exactly to its base
        self.reset()

    def reset(self):
        self.x = 0.0
        self.x_dot = 0.0
        theta_deg = random.uniform(self.cfg.theta_init_range_deg[0], self.cfg.theta_init_range_deg[1])
        self.theta = math.radians(theta_deg)
        self.theta_dot = 0.0
        self.force = 0.0

    def set_force(self, f: float):
        self.force = float(f)

    def step(self, dt: float):
        # Standard cart-pole continuous dynamics (as used in OpenAI Gym)
        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)

        temp = (self.force + self.poleml * (self.theta_dot * self.theta_dot) * sintheta) / self.total_m
        thetaacc = (self.cfg.g * sintheta - costheta * temp) / (
            self.pole_length * (4.0 / 3.0 - self.cfg.m_p * (costheta * costheta) / self.total_m)
        )
        xacc = temp - (self.poleml * thetaacc * costheta) / self.total_m

        # Semi-implicit Euler with light exponential damping for stability
        self.x_dot += dt * xacc
        self.theta_dot += dt * thetaacc
        if self.cfg.linear_damping > 0.0 or self.cfg.angular_damping > 0.0:
            lin_damp = math.exp(-self.cfg.linear_damping * dt)
            ang_damp = math.exp(-self.cfg.angular_damping * dt)
            self.x_dot *= lin_damp
            self.theta_dot *= ang_damp

        self.x += dt * self.x_dot
        self.theta += dt * self.theta_dot

    def should_reset(self) -> bool:
        # Leaving the edges of the rail
        half_rail = self.cfg.rail_size[0] * 0.5
        half_cart = self.cfg.cart_size[0] * 0.5
        if abs(self.x) > half_rail - half_cart:
            return True
        
        # The pole is falling (beyond angle threshold)
        if abs(math.degrees(self.theta)) > self.cfg.theta_threshold_deg:
            return True
        return False

    def compute_reward(self) -> float:
        return 0.0 if self.should_reset() else 1.0

class CartPoleRenderer(ShowBase):
    def __init__(self, cfg: CartPoleConfig, logic: CarPoleLogic):
        loadPrcFileData('', f'show-frame-rate-meter {1 if cfg.show_fps else 0}\n')
        loadPrcFileData('', f'sync-video 0\n')
        loadPrcFileData('', f'win-size {cfg.width} {cfg.height}\n')

        super().__init__()

        self.cfg = cfg
        self.logic = logic

        self.disableMouse() # allows for static camera

        # Input
        self._key_left = False
        self._key_right = False
        self.accept('arrow_left', self._on_key, ['left', True])
        self.accept('arrow_left-up', self._on_key, ['left', False])
        self.accept('arrow_right', self._on_key, ['right', True])
        self.accept('arrow_right-up', self._on_key, ['right', False])

        # Scene
        self._build_scene()

        # Update
        self.taskMgr.add(self._update, 'update')

    def _build_scene(self):
        # Cart
        self.cart_np = self.loader.loadModel('models/box')
        self.cart_np.reparentTo(self.render)
        self.cart_np.setColor(*self.cfg.cart_color)
        self.cart_np.setScale(*self.cfg.cart_size)
        self.cart_np.setTextureOff(1)
        bake_pivot_to_rel(self.cart_np, (0.5, 0.5, 0.5))

        # Hinge from the side of the cart
        self.hinge_np = self.cart_np.attachNewNode('hinge')
        self.hinge_np.setPos(0.0,
        -0.5 *(self.cfg.cart_size[1] + self.cfg.pole_size[1]),
        0.0)

        # Pole (child of hinge). Translate up by half its length so base sits on hinge
        self.pole_np = self.loader.loadModel('models/box')
        self.pole_np.reparentTo(self.hinge_np)
        self.pole_np.setColor(*self.cfg.pole_color)
        self.pole_np.setScale(*self.cfg.pole_size)
        self.pole_np.setTextureOff(1)
        bake_pivot_to_rel(self.pole_np, (0.5, 0.5, 0.5))
        self.pole_np.setPos(0.0, 0.0, self.cfg.pole_size[2] * 0.45) # middle of the hinge is near the base center

        # Rail (visual reference)
        self.rail_np = self.loader.loadModel('models/box')
        self.rail_np.reparentTo(self.render)
        self.rail_np.setColor(*self.cfg.rail_color)
        self.rail_np.setScale(*self.cfg.rail_size)
        self.rail_np.setTextureOff(1)
        bake_pivot_to_rel(self.rail_np, (0.5, 0.5, 0.5))
        self.rail_np.setPos(0.0, 0.0, 0.0)

        # Camera initial placement
        self._update_camera()

        self._add_lights()

    def _add_lights(self):
        alight = AmbientLight('ambient')
        alight.setColor((0.25, 0.25, 0.28, 1.0))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        dlight = DirectionalLight('sun')
        dlight.setColor((0.95, 0.95, 0.95, 1.0))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.lookAt(Vec3(-0.5, -1.0, -1.5))
        self.render.setLight(dlnp)

    def _on_key(self, which: str, down: bool):
        if which == 'left':
            self._key_left = down
        elif which == 'right':
            self._key_right = down

    def _apply_input(self):
        f = 0.0
        if self._key_left and not self._key_right:
            f = -self.cfg.force_mag
        elif self._key_right and not self._key_left:
            f = self.cfg.force_mag
        self.logic.set_force(f)

    def _update_camera(self):
        self.camera.setPos(*self.cfg.cam_position)
        self.camera.lookAt(self.cfg.cam_look_at)

    def _update_transforms(self):
        # Cart translation along X
        self.cart_np.setPos(self.logic.x, 0.0, 0.0)
        # Pole rotation about hinge around world Y (cart local Y)
        self.hinge_np.setHpr(0.0, 0.0, math.degrees(self.logic.theta))

    def _update(self, task):
        # Fixed-step substepping for stability
        dt = max(1e-3, min(1.0 / 30.0, globalClock.getDt()))
        steps = max(1, int(round(dt / self.cfg.dt)))
        sub_dt = dt / steps
        for _ in range(steps):
            self._apply_input()
            self.logic.step(sub_dt)

        self._update_transforms()
        if self.logic.should_reset():
            self.logic.reset()
        return task.cont


if __name__ == '__main__':
    cfg = CartPoleConfig()
    logic = CarPoleLogic(cfg)
    app = CartPoleRenderer(cfg, logic)
    app.run()