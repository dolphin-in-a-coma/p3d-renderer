import numpy as np

from panda3d.core import loadPrcFileData

try:
    from .renderer.renderer import P3DRenderer
    from .renderer.shader_context import P3DShaderContext
except ImportError:
    # Fallback when run as a script (no package parent)
    from renderer.renderer import P3DRenderer
    from renderer.shader_context import P3DShaderContext

class CartPoleDemo(P3DRenderer):
    def __init__(self):
        num_scenes = 1024*16 # 1024*4 # 1024*4 # 1024*4 # ~1024 is the best on Mac, ~8k is the best on L4
        tile_resolution = (64,64)
        super().__init__(tile_resolution=tile_resolution, num_scenes=num_scenes, offscreen=True, interactive=False)

        instances_per_scene = 1

        # print(self.cfg)
        # self.cam.setPos(0, -50, 10)
        # self.cam.lookAt(0, 0, 0)

        self.rail_size = (6.0, 0.05, 0.05)
        self.cart_size = (1.2, 0.8, 0.5)
        self.pole_size = (0.1, 0.1, 2.0)

        self.rail_pos_color = (0.2, 0.2, 0.2, 1.0)
        self.cart_pos_color_range = ((0.6, 0.8, 1.0, 1.0), (1.0, 0.6, 0.8, 1.0))
        self.pole_pos_color = (1.0, 0.7, 0.2, 1.0)

        # Add nodes and camera via renderer helpers
        self.rail  = self.add_node('models/box', model_pivot_relative_point=(0.5, 0.5, 0.5),
        model_scale=self.rail_size, instances_per_scene=instances_per_scene, shared_across_scenes=True)
        self.cart  = self.add_node('models/box', model_pivot_relative_point=(0.5, 0.5, 0.5), model_scale=self.cart_size, instances_per_scene=instances_per_scene, shared_across_scenes=False)
        self.pole  = self.add_node('models/box', model_pivot_relative_point=(0.5, 0.5, 0.05), model_scale=self.pole_size, instances_per_scene=instances_per_scene, shared_across_scenes=False)

        # Set positions
        self.rail_pos = np.zeros((1, instances_per_scene, 3), np.float32)
        self.cart_pos = np.zeros((num_scenes, instances_per_scene, 3), np.float32)
        self.pole_pos = self.cart_pos.copy() + np.array([0, (self.cart_size[1] + self.pole_size[1]) * 0.5, 0])
        
        self.rail.set_positions(self.rail_pos)
        self.cart.set_positions(self.cart_pos)
        self.pole.set_positions(self.pole_pos)

        # Set colors
        self.rail_base_color = np.ones((1, instances_per_scene, 4), np.float32) * np.array(self.rail_pos_color)
        self.cart_base_color = np.linspace(self.cart_pos_color_range[0], self.cart_pos_color_range[1], num_scenes).astype(np.float32)
        self.pole_base_color = np.ones((num_scenes, instances_per_scene, 4), np.float32) * np.array(self.pole_pos_color)

        self.rail.set_colors(self.rail_base_color)
        self.cart.set_colors(self.cart_base_color)
        self.pole.set_colors(self.pole_base_color)

        self.pole_hpr = np.zeros((num_scenes, instances_per_scene, 3), np.float32)
        self.pole_hpr[:, :, 1] = np.pi * 0.5
        self.pole.set_hprs(self.pole_hpr)

        self.cart_x_pos = self.cart_pos[:, :, 0:1].copy()
        self.pole_theta = np.zeros_like(self.cart_x_pos)

        self.add_camera()

        # Global lighting controller
        self.light = self.add_light(ambient=(0.2, 0.2, 0.25), dir_dir=(0.4, -0.6, -0.7), dir_col=(1,1,1), strength=1.0)

        if self.cfg.interactive:
            self.init_interactive_controls()

        # Setup environment
        self.setup_environment()


    def init_interactive_controls(self):

        # Interactive controls for demo (arrow keys for cart, Q/E for pole tilt)
        self._ctrl = { 'left': False, 'right': False, 'tilt_left': False, 'tilt_right': False }
        def _set_ctrl(k, v):
            self._ctrl[k] = v
        self.accept('arrow_left', _set_ctrl, ['left', True])
        self.accept('arrow_left-up', _set_ctrl, ['left', False])
        self.accept('arrow_right', _set_ctrl, ['right', True])
        self.accept('arrow_right-up', _set_ctrl, ['right', False])
        self.accept('q', _set_ctrl, ['tilt_left', True])
        self.accept('q-up', _set_ctrl, ['tilt_left', False])
        self.accept('e', _set_ctrl, ['tilt_right', True])
        self.accept('e-up', _set_ctrl, ['tilt_right', False])

    def _step(self, cart_x_delta=None, pole_theta_delta=None):
        if cart_x_delta is None:
            cart_x_delta = np.zeros((self.num_scenes, self.cart.instances_per_scene, 1), dtype=np.float32)
        else:
            cart_x_delta = np.asarray(cart_x_delta, dtype=np.float32)

        self.cart_x_pos = self.cart_x_pos + cart_x_delta
        self.cart_x_pos = np.clip(self.cart_x_pos, 
                                 -self.rail_size[0] * 0.5 + self.cart_size[0] * 0.5,
                                 self.rail_size[0] * 0.5 - self.cart_size[0] * 0.5)

        self.cart_pos[:, :, 0:1] = self.cart_x_pos
        self.cart.set_positions(self.cart_pos)

        self.pole_pos[:, :, 0:1] = self.cart_x_pos
        self.pole.set_positions(self.pole_pos, lazy=True) # don't upload immediately, wait for a not lazy step

        if pole_theta_delta is None:
            pole_theta_delta = np.zeros((self.num_scenes, self.pole.instances_per_scene, 1), dtype=np.float32)
        else:
            pole_theta_delta = np.asarray(pole_theta_delta, dtype=np.float32)

        self.pole_theta = self.pole_theta + pole_theta_delta
        self.pole_theta = np.clip(self.pole_theta, -np.pi, np.pi)
        self.pole_hpr[:, :, 1:2] = self.pole_theta
        self.pole.set_hprs(self.pole_hpr)

    def _interactive_step(self):
        # Build deltas from current key states
        move = (1.0 if self._ctrl['right'] else 0.0) - (1.0 if self._ctrl['left'] else 0.0)
        tilt = (1.0 if self._ctrl['tilt_right'] else 0.0) - (1.0 if self._ctrl['tilt_left'] else 0.0)
        # Scale by dt for smoothness
        dx = 2.0 * float(self.cfg.dt) * move
        dtheta = 1.5 * float(self.cfg.dt) * tilt
        cart_x_delta = np.full((self.num_scenes, self.cart.instances_per_scene, 1), dx, dtype=np.float32)
        pole_theta_delta = np.full((self.num_scenes, self.pole.instances_per_scene, 1), dtheta, dtype=np.float32)
        self._step(cart_x_delta, pole_theta_delta)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    app = CartPoleDemo()
    num_steps = 10000
    plot_every = -100
    imgs = []

    # num_deltas = 99
    # cart_x_deltas_list = [(np.random.rand(app.num_scenes, app.cart.instances_per_scene, 1) * 0.1 - 0.05).astype(np.float32) for _ in range(num_deltas)]
    # pole_theta_deltas_list = [(np.random.rand(app.num_scenes, app.pole.instances_per_scene, 1) * 0.1 - 0.05).astype(np.float32) for _ in range(num_deltas)]
    for i in range(num_steps):

        # i_delta = i % num_deltas
        # Example: create random deltas outside update_instances and apply per step
        cart_x_delta = (np.random.rand(app.num_scenes, app.cart.instances_per_scene, 1) * 0.1 - 0.05).astype(np.float32)
        pole_theta_delta = (np.random.rand(app.num_scenes, app.pole.instances_per_scene, 1) * 0.1 - 0.05).astype(np.float32)
        img = app.step(cart_x_delta, pole_theta_delta)
        if plot_every > 0 and i % plot_every == 0:
            plt.figure(figsize=(10, 10))
            plt.imshow(img.cpu())
            plt.savefig(f'img_cartpole_{i}.png')
            plt.show()
            plt.close()