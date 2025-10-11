import math
from dataclasses import dataclass

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
    height: int = 480
    show_fps: bool = True

    # Physics
    dt: float = 1 / 120
    g: float = 9.8
    m_c: float = 1.0
    m_p: float = 0.1
    length: float = 0.5  # half pole length used by the standard equations
    force_mag: float = 10.0
    linear_damping: float = 0.5
    angular_damping: float = 0.2

    # Visual scales
    cart_size: tuple[float, float, float] = (1.2, 0.8, 0.5)
    pole_size: tuple[float, float, float] = (0.1, 0.1, 2.0)

    # Camera follow
    cam_left_angle_deg: float = 30.0
    cam_distance: float = 6.0
    cam_height: float = 1.8


class CarPoleLogic:
    def __init__(self, cfg: CartPoleConfig):
        self.cfg = cfg
        self.total_m = cfg.m_c + cfg.m_p
        self.poleml = cfg.m_p * cfg.length
        self.reset()

    def reset(self):
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
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
            self.cfg.length * (4.0 / 3.0 - self.cfg.m_p * (costheta * costheta) / self.total_m)
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


class CartPoleRenderer(ShowBase):
    def __init__(self, cfg: CartPoleConfig, logic: CarPoleLogic):
        loadPrcFileData('', f'show-frame-rate-meter {1 if cfg.show_fps else 0}\n')
        loadPrcFileData('', f'sync-video 0\n')
        loadPrcFileData('', f'win-size {cfg.width} {cfg.height}\n')

        super().__init__()

        self.cfg = cfg
        self.logic = logic

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
        self.cart_np.setColor(0.6, 0.8, 1.0, 1.0)
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
        self.pole_np.setColor(1.0, 0.7, 0.2, 1.0)
        self.pole_np.setScale(*self.cfg.pole_size)
        self.pole_np.setTextureOff(1)
        bake_pivot_to_rel(self.pole_np, (0.5, 0.5, 0.5))
        self.pole_np.setPos(0.0, 0.0, self.cfg.pole_size[2] * 0.45) # middle of the hinge is near the base center

        # Rail (visual reference)
        self.rail_np = self.loader.loadModel('models/box')
        self.rail_np.reparentTo(self.render)
        self.rail_np.setColor(0.2, 0.2, 0.2, 1.0)
        self.rail_np.setScale(6.0, 0.05, 0.05)
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
        # Follow the cart from a fixed left angle
        ang = math.radians(self.cfg.cam_left_angle_deg)
        offset_x = -math.sin(ang) * self.cfg.cam_distance
        offset_y = -math.cos(ang) * self.cfg.cam_distance
        target_x = self.logic.x
        target_y = 0.0
        target_z = 0.0
        self.camera.setPos(target_x + offset_x, target_y + offset_y, target_z + self.cfg.cam_height)
        self.camera.lookAt(target_x, target_y, target_z + self.cfg.cart_size[2] * 0.5)

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
        self._update_camera()
        return task.cont


if __name__ == '__main__':
    cfg = CartPoleConfig()
    logic = CarPoleLogic(cfg)
    app = CartPoleRenderer(cfg, logic)
    app.run()


