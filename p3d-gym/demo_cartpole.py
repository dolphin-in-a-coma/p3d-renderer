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
        num_scenes = 1024*4 # ~1024 is the best on Mac, ~8k is the best on L4
        instances_per_scene = 1
        super().__init__(tile_resolution=(64,64), num_scenes=num_scenes, offscreen=True)

        gsg = self.win.getGsg()
        print("Pipe:", self.win.pipe.getType().getName())      # expect: eglGraphicsPipe
        print("Renderer:", gsg.getDriverRenderer())        # e.g., "NVIDIA L4/PCIe/SSE2"


        print(self.cfg)
        self.cam.setPos(0, -50, 10)
        self.cam.lookAt(0, 0, 0)
        self.disableMouse()

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
        target_x = np.linspace(-self.rail_size[0] * 0.5 + self.cart_size[0] * 0.5, self.rail_size[0] * 0.5 - self.cart_size[0] * 0.5, num_scenes)
        target_y = np.zeros_like(target_x)
        target_z = np.zeros_like(target_x)
        target_k3 = np.stack([target_x, target_y, target_z], axis=1).astype(np.float32)
        self._p3d_cam.look_at(target_k3=target_k3)

        # self._set_tiles_auto()
        self.start()

        # Global lighting controller
        self.light = self.add_light(ambient=(0.2, 0.2, 0.25), dir_dir=(0.4, -0.6, -0.7), dir_col=(1,1,1), strength=1.0)

        # self.taskMgr.add(self.update_camera, 'update_camera')
        self.taskMgr.add(self.update_instances, 'update_instances')

        self._setup_offscreen_rt()

        self._warmup()

        self.taskMgr.doMethodLater(0, self._init_frame_grabber_once, 'init-frame-grabber-once')
        # TODO: move to some post-init method

    def update_instances(self, task):
        cart_x_delta = np.random.rand(self.num_scenes, self.cart.instances_per_scene, 1) * 0.1 - 0.05
        self.cart_x_pos = self.cart_x_pos + cart_x_delta
        self.cart_x_pos = np.clip(self.cart_x_pos, 
                                 -self.rail_size[0] * 0.5 + self.cart_size[0] * 0.5,
                                 self.rail_size[0] * 0.5 - self.cart_size[0] * 0.5)

        self.cart_pos[:, :, 0:1] = self.cart_x_pos
        self.cart.set_positions(self.cart_pos)

        self.pole_pos[:, :, 0:1] = self.cart_x_pos
        self.pole.set_positions(self.pole_pos)

        pole_theta_delta = np.random.rand(self.num_scenes, self.pole.instances_per_scene, 1) * 0.1 - 0.05
        self.pole_theta = self.pole_theta + pole_theta_delta
        self.pole_theta = np.clip(self.pole_theta, -np.pi, np.pi)
        self.pole_hpr[:, :, 1:2] = self.pole_theta
        self.pole.set_hprs(self.pole_hpr)

        # print(self.cart_pos.shape)
        return task.cont


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    app = CartPoleDemo()
    num_steps = 10000
    plot_every = -1
    imgs = []
    for i in range(num_steps):
        img = app.step_and_grab()
        if plot_every > 0 and i % plot_every == 0:
            plt.figure(figsize=(10, 10))
            plt.imshow(img.cpu())
            plt.savefig(f'img_cartpole_{i}.png')
            plt.show()
            plt.close()

