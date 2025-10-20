import numpy as np

from panda3d.core import loadPrcFileData

try:
    from .renderer.renderer import P3DRenderer
    from .renderer.shader_context import P3DShaderContext
except ImportError:
    # Fallback when run as a script (no package parent)
    from renderer.renderer import P3DRenderer
    from renderer.shader_context import P3DShaderContext


# # Set GL version and FPS meter
# loadPrcFileData('', 'gl-version 3 2')
# loadPrcFileData('', 'show-frame-rate-meter 1\n')

class Demo(P3DRenderer):
    def __init__(self):
        num_scenes = 128
        instances_per_scene = 3
        super().__init__(num_scenes=num_scenes, offscreen=True)
        self.cam.setPos(0, -50, 10)
        self.cam.lookAt(0, 0, 0)
        self.disableMouse()
        
        # Add nodes and camera via renderer helpers
        self.node  = self.add_node('models/smiley', instances_per_scene=instances_per_scene, shared_across_scenes=True)
        self.node2 = self.add_node('models/smiley', instances_per_scene=instances_per_scene, shared_across_scenes=False)

        self.add_camera()

        # Set positions
        positions = np.random.rand(num_scenes, instances_per_scene, 3) * 30 - 15
        self.node.set_positions(positions)
        self.node2.set_positions(positions)
        # Flattened counts for animation
        self.B = self.node.total_instances
        self.base_pos = positions.reshape(self.B, 3).astype(np.float32)
        rng = np.random.default_rng(123)
        self.vel = rng.uniform(-6, 6, size=(self.B, 3)).astype(np.float32)
        self.scale_base = rng.uniform(0.5, 1.8, size=(self.B, 1)).astype(np.float32)
        self.scale_amp = rng.uniform(0.2, 0.8, size=(self.B, 1)).astype(np.float32)
        self.phase = rng.uniform(0.0, 2*np.pi, size=(self.B, 3)).astype(np.float32)

        # Static random rotations per-instance (H, P, R in radians)
        self.rot_hpr = rng.uniform(-np.pi, np.pi, size=(self.B, 3)).astype(np.float32)
        self.R3 = P3DShaderContext._rotation_mats_from_hpr(self.rot_hpr)
        # Angular velocities for dynamic rotation (radians/sec per axis)
        self.ang_vel = rng.uniform(-0.8, 0.8, size=(self.B, 3)).astype(np.float32)
        
        # Set colors
        colors = np.random.rand(num_scenes, instances_per_scene, 4).astype(np.float32)
        colors[..., 3] = 1.0 # full alpha
        self.node.set_colors(colors)
        self.node2.set_colors(colors)

        # Camera + tiles via renderer
        self._set_tiles_auto()
        self.start()

        # Global lighting controller
        self.light = self.add_light(ambient=(0.2, 0.2, 0.25), dir_dir=(0.4, -0.6, -0.7), dir_col=(1,1,1), strength=1.0)

        # self.taskMgr.add(self.update_camera, 'update_camera')
        self.taskMgr.add(self.update_instances, 'update_instances')

        self._warmup()

        self.taskMgr.doMethodLater(0, self._init_frame_grabber_once, 'init-frame-grabber-once')

    def update_instances(self, task):
        from direct.showbase.ShowBaseGlobal import globalClock
        t = float(globalClock.getFrameTime())

        # Positions: oscillate along random velocity directions
        pos = self.base_pos + self.vel * np.sin(t * 0.7 + self.phase[:, 0:1])

        # Scales: pulsate uniformly per instance
        s = self.scale_base + self.scale_amp * np.sin(t * 1.3 + self.phase[:, 1:2])
        s = np.clip(s, 0.2, 3.0)

        # Build transform matrices (dynamic rotation * uniform scale + translation)
        hpr_t = (self.rot_hpr + self.ang_vel * t).astype(np.float32, copy=False)
        R3_t = P3DShaderContext._rotation_mats_from_hpr(hpr_t)
        mats = np.zeros((self.B, 4, 4), np.float32)
        mats[:, 3, 3] = 1.0
        mats[:, 0:3, 0:3] = R3_t * s[:, 0][:, None, None]
        mats[:, 0:3, 3] = pos
        if getattr(self.node, 'shared_across', False):
            self.node.set_transforms(mats[:self.node.instances_per_scene])
        else:
            self.node.set_transforms(mats)
        if getattr(self.node2, 'shared_across', False):
            self.node2.set_transforms(mats[:self.node2.instances_per_scene])
        else:
            self.node2.set_transforms(mats)

        # Colors: smooth cycling per channel
        col_rgb = 0.5 + 0.5 * np.sin(t * np.array([0.9, 1.1, 1.3], np.float32) + self.phase)
        col_rgb = col_rgb.astype(np.float32)
        col = np.concatenate([col_rgb, np.ones((self.B, 1), np.float32)], axis=1)

        if getattr(self.node, 'shared_across', False):
            self.node.set_colors(col[:self.node.instances_per_scene])
        else:
            self.node.set_colors(col)
        if getattr(self.node2, 'shared_across', False):
            self.node2.set_colors(col[:self.node2.instances_per_scene])
        else:
            self.node2.set_colors(col)

        return task.cont

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    app = Demo()
    num_steps = 100000
    plot_every = -1
    imgs = []
    for i in range(num_steps):
        #img = app.taskMgr.step()
        # print("stepping...", i)
        img = app.step_and_grab()
        if plot_every > 0 and i % plot_every == 0:
            imgs.append(img)
        # print(img.shape)


            plt.figure(figsize=(10, 10))
            # for i, img in enumerate(imgs):
            #     plt.subplot(10, 10, i + 1)
            #     plt.imshow(img.cpu())
            #     plt.axis('off')

            plt.imshow(img.cpu())


            plt.savefig('imgs.png')
            plt.show()
            plt.close()