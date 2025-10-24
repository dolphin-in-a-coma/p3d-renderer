import time

import math
from dataclasses import dataclass
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data.tensor_specs import Composite, Unbounded, Categorical

try:
    from .renderer.renderer import P3DRenderer
    from .config import P3DConfig
except ImportError:
    # Fallback when run as a script (no package parent)
    from renderer.renderer import P3DRenderer
    from config import P3DConfig


@dataclass
class CartPoleConfig(P3DConfig):
    # TorchRL env defaults
    direct_obs_dim: int | None = 4
    action_n: int | None = 2
    action_type: str = 'discrete'
    max_steps: int = 500
    auto_reset: bool = True

    # Rendering defaults
    num_channels: int = 3
    tile_resolution: tuple[int, int] | None = (64, 64)
    offscreen: bool = True
    report_fps: bool = False

    # CartPole physics defaults
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5          # half-pole length
    force_mag: float = 10.0
    tau: float = 0.02            # s between updates
    theta_threshold_deg: float = 12.0
    x_threshold: float = 2.4


class CartPoleRenderer(P3DRenderer):
    def __init__(self, cfg: P3DConfig | dict | None = None):
        super().__init__(cfg)
        instances_per_scene = 1
        # Geometry sizes/colors
        self.rail_size = (6.0, 0.05, 0.05)
        self.cart_size = (1.2, 0.8, 0.5)
        self.pole_size = (0.1, 0.1, 2.0)
        self.rail_pos_color = (0.2, 0.2, 0.2, 1.0)
        self.cart_pos_color_range = ((0.6, 0.8, 1.0, 1.0), (1.0, 0.6, 0.8, 1.0))
        self.pole_pos_color = (1.0, 0.7, 0.2, 1.0)

        # Nodes
        self.rail  = self.add_node('models/box', model_pivot_relative_point=(0.5, 0.5, 0.5),
                                    model_scale=self.rail_size, instances_per_scene=instances_per_scene, shared_across_scenes=True)
        self.cart  = self.add_node('models/box', model_pivot_relative_point=(0.5, 0.5, 0.5), model_scale=self.cart_size, instances_per_scene=instances_per_scene, shared_across_scenes=False)
        self.pole  = self.add_node('models/box', model_pivot_relative_point=(0.5, 0.5, 0.05), model_scale=self.pole_size, instances_per_scene=instances_per_scene, shared_across_scenes=False)

        # Position/color buffers
        num_scenes = int(self.cfg.num_scenes)
        self.rail_pos = np.zeros((1, instances_per_scene, 3), np.float32)
        self.cart_pos = np.zeros((num_scenes, instances_per_scene, 3), np.float32)
        self.pole_pos = self.cart_pos.copy() + np.array([0, (self.cart_size[1] + self.pole_size[1]) * 0.5, 0], dtype=np.float32)
        self.rail.set_positions(self.rail_pos)
        self.cart.set_positions(self.cart_pos)
        self.pole.set_positions(self.pole_pos)

        self.rail_base_color = np.ones((1, instances_per_scene, 4), np.float32) * np.array(self.rail_pos_color, dtype=np.float32)
        self.cart_base_color = np.linspace(self.cart_pos_color_range[0], self.cart_pos_color_range[1], num_scenes).astype(np.float32)
        self.pole_base_color = np.ones((num_scenes, instances_per_scene, 4), np.float32) * np.array(self.pole_pos_color, dtype=np.float32)
        self.rail.set_colors(self.rail_base_color)
        self.cart.set_colors(self.cart_base_color)
        self.pole.set_colors(self.pole_base_color)

        self.pole_hpr = np.zeros((num_scenes, instances_per_scene, 3), np.float32)
        self.pole_hpr[:, :, 1] = np.pi * 0.5
        self.pole.set_hprs(self.pole_hpr)

        # Buffers mapped from state
        self.cart_x_pos = self.cart_pos[:, :, 0:1].copy()
        self.pole_theta = np.zeros_like(self.cart_x_pos)

        self.add_camera()
        self.add_light()
        self.setup_environment()

    def _step(self, state_batch: torch.Tensor | np.ndarray | None = None):
        # Accept direct state [B, 4] where x and theta are at indices 0 and 2
        if state_batch is None:
            return
        if isinstance(state_batch, torch.Tensor):
            state_np = state_batch.detach().cpu().numpy()
        else:
            state_np = np.asarray(state_batch)
        B = state_np.shape[0]
        # Truncate/expand to num_scenes
        N = int(self.cfg.num_scenes)
        if B != N:
            if B < N:
                pad = np.repeat(state_np[-1:], N - B, axis=0)
                state_np = np.concatenate([state_np, pad], axis=0)
            else:
                state_np = state_np[:N]
        x = state_np[:, 0:1]
        theta = state_np[:, 2:3]
        self.cart_x_pos[:, :, 0] = x
        self.cart_pos[:, :, 0:1] = self.cart_x_pos
        self.cart.set_positions(self.cart_pos)
        self.pole_pos[:, :, 0:1] = self.cart_x_pos
        self.pole.set_positions(self.pole_pos, lazy=True)
        self.pole_theta[:, :, 0] = theta
        self.pole_hpr[:, :, 1:2] = self.pole_theta
        self.pole.set_hprs(self.pole_hpr)


class CartPoleEnv(EnvBase):
    def __init__(self, device: str | torch.device = "cpu", batch_size: torch.Size = torch.Size([128]), seed: int | None = None, render: bool = False, p3d_cfg: P3DConfig | dict | None = None):
        super().__init__(device=torch.device(device), batch_size=batch_size)
        self._renderer: CartPoleRenderer | None = None
        cfg = None
        if render:
            base_cfg = p3d_cfg if p3d_cfg is not None else {}
            if isinstance(base_cfg, dict):
                cfg = CartPoleConfig.from_config(base_cfg)
            elif isinstance(base_cfg, P3DConfig):
                cfg = CartPoleConfig.from_config(base_cfg.__dict__)
            else:
                cfg = CartPoleConfig()
            # tiles/num_scenes from batch
            N = 1
            for d in (batch_size if batch_size != torch.Size([]) else torch.Size([1])):
                N *= int(d)
            cols = int(math.ceil(math.sqrt(N)))
            rows = int(math.ceil(N / cols))
            cfg.tiles = (cols, rows)
            cfg.num_scenes = N
            if cfg.tile_resolution is None and cfg.window_resolution is None:
                cfg.tile_resolution = (64, 64)
            cfg.offscreen = True
            cfg.report_fps = False
            self._renderer = CartPoleRenderer(cfg)

        # Physics params
        self.gravity = float(cfg.gravity if cfg is not None else 9.8)
        self.masscart = float(cfg.masscart if cfg is not None else 1.0)
        self.masspole = float(cfg.masspole if cfg is not None else 0.1)
        self.total_mass = self.masscart + self.masspole
        self.length = float(cfg.length if cfg is not None else 0.5)
        self.polemass_length = self.masspole * self.length
        self.force_mag = float(cfg.force_mag if cfg is not None else 10.0)
        self.tau = float(cfg.tau if cfg is not None else 0.02)
        self.theta_threshold = float((cfg.theta_threshold_deg if cfg is not None else 12.0) * 2 * math.pi / 360.0)
        self.x_threshold = float(cfg.x_threshold if cfg is not None else 2.4)
        self.max_steps = int(cfg.max_steps if cfg is not None else 500)
        self.auto_reset = bool(cfg.auto_reset if cfg is not None else True)

        # Specs
        bs = self.batch_size if self.batch_size != torch.Size([]) else torch.Size([1])
        obs_fields = {
            "observation": Unbounded(shape=bs + torch.Size([4]), dtype=torch.float32, device=self.device),
        }
        if self._renderer is not None:
            C = int(self._renderer.cfg.num_channels)
            W = int(self._renderer.cfg.tile_resolution[0])
            H = int(self._renderer.cfg.tile_resolution[1])
            obs_fields["pixels"] = Unbounded(shape=bs + torch.Size([C, H, W]), dtype=torch.uint8, device=self.device)
        self.observation_spec = Composite(**obs_fields, shape=bs)
        self.action_spec = Categorical(n=2, shape=bs, dtype=torch.long, device=self.device)
        self.reward_spec = Unbounded(shape=bs + torch.Size([1]), dtype=torch.float32, device=self.device)
        self.done_spec = Unbounded(shape=bs + torch.Size([1]), dtype=torch.bool, device=self.device)

        if seed is not None:
            self.set_seed(seed)

    # ---- Dynamics / termination / reward hooks ----
    def _dynamics(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, x_dot, theta, theta_dot = obs.unbind(-1)
        force = torch.where(action == 1, self.force_mag, -self.force_mag).to(torch.float32)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        temp = (force + self.polemass_length * theta_dot.pow(2) * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta.pow(2) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return torch.stack([x, x_dot, theta, theta_dot], dim=-1)

    def _termination(self, obs: torch.Tensor, step_count: torch.Tensor) -> torch.Tensor:
        x, x_dot, theta, theta_dot = obs.unbind(-1)
        done = (
            (x.abs() > self.x_threshold)
            | (theta.abs() > self.theta_threshold)
            | (step_count >= (self.max_steps - 1))
        ).unsqueeze(-1)
        return done

    def _reward(self, obs: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(done, dtype=torch.float32, device=self.device)

    # ---- TorchRL API ----
    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(int(seed))

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        bs = self.batch_size if self.batch_size != torch.Size([]) else torch.Size([1])
        state = torch.empty(*bs, 4, dtype=torch.float32, device=self.device).uniform_(-0.05, 0.05)
        step_count = torch.zeros(*bs, dtype=torch.long, device=self.device)
        done = torch.zeros(*bs, 1, dtype=torch.bool, device=self.device)
        td_fields = {
            "observation": state,
            "step_count": step_count,
            "done": done,
        }
        if self._renderer is not None:
            try:
                pixels = self._renderer.step(state)
                td_fields["pixels"] = pixels
            except Exception:
                pass
        return TensorDict(td_fields, batch_size=self.batch_size, device=self.device)

    @torch.no_grad()
    def _step(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict.get("observation", None)
        if obs is None:
            td0 = self._reset()
            obs = td0["observation"]
            step_count = td0["step_count"]
        else:
            step_count = tensordict.get(
                "step_count",
                torch.zeros_like(obs[..., 0], dtype=torch.long, device=self.device),
            )

        action = tensordict["action"].to(self.device)
        next_obs_raw = self._dynamics(obs, action)
        done = self._termination(next_obs_raw, step_count)
        reward = self._reward(next_obs_raw, done)

        pixels = None
        if self._renderer is not None:
            try:
                pixels = self._renderer.step(next_obs_raw)
            except Exception:
                pixels = None

        if self.auto_reset:
            reset_state = torch.empty_like(next_obs_raw).uniform_(-0.05, 0.05)
            next_obs = torch.where(done, reset_state, next_obs_raw)
            next_step_count = torch.where(done.squeeze(-1), torch.zeros_like(step_count), step_count + 1)
        else:
            next_obs = next_obs_raw
            next_step_count = step_count + 1

        next_fields = {
            "observation": next_obs,
            "reward": reward,
            "done": done,
            "step_count": next_step_count,
        }
        if pixels is not None:
            next_fields["pixels"] = pixels

        return TensorDict({
            "next": next_fields,
            "reward": reward,
            "done": done,
        }, batch_size=self.batch_size, device=self.device)

# Quick test
if __name__ == "__main__":
    num_scenes = 1024*8 #256
    tile_resolution = (64, 64)
    config = CartPoleConfig(num_scenes=num_scenes, tile_resolution=tile_resolution)
    # renderer = CartPoleRenderer(config)
    # renderer.setup_environment()
    env = CartPoleEnv(device="cpu",
    batch_size=torch.Size([num_scenes]),
    p3d_cfg=config,
    render=True)
    td = env.reset()
    total = 0.0
    tm_start = time.time()
    steps = 1000
    total_frames = steps * num_scenes

    for t in range(steps):
        a = env.action_spec.rand()
        td["action"] = a
        td = env.step(td)
        total += td["next", "reward"].sum().item()
        td = td["next"]
        # print(td)
        # break
    tm_end = time.time()
    fps = total_frames / (tm_end - tm_start)
    print(f"Time taken: {tm_end - tm_start} seconds")
    print(f"FPS: {fps}")