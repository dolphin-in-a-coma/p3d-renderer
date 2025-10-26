import time

import math
from dataclasses import dataclass
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

try:
    from .renderer.renderer import P3DRenderer
    from .config import P3DConfig
    from .env import P3DEnv
except ImportError:
    # Fallback when run as a script (no package parent)
    from renderer.renderer import P3DRenderer
    from config import P3DConfig
    from env import P3DEnv


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
    theta_threshold_deg: float = 90.0
    x_threshold: float = 2.4

    seed: int = 0
    render: bool = True
    # Initial state ranges
    init_x_range: tuple[float, float] = (-2, 2)
    # Theta ranges are specified in DEGREES in config; converted to radians in env init
    init_theta_range_deg: tuple[float, float] = (-30, 30)
    init_x_dot_range: tuple[float, float] = (-1, 1)
    # Theta-dot range specified in DEGREES/SEC in config; converted to radians/sec in env init
    init_theta_dot_range_deg: tuple[float, float] = (-15, 15)
    # Saving controls
    save_every_steps: int = 50  # < 0 disables saving; 0 saves every step; >0 saves every N steps
    save_examples_num: int = 16
    save_out_dir: str | None = None

    # Parallel env controls
    worker_index: int = 0
    num_workers: int = 1

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
        self.rail_pos = torch.zeros((1, instances_per_scene, 3), dtype=torch.float32)
        self.cart_pos = torch.zeros((num_scenes, instances_per_scene, 3), dtype=torch.float32)
        self.pole_pos = self.cart_pos.clone() + torch.tensor([0, (self.cart_size[1] + self.pole_size[1]) * 0.5, 0], dtype=torch.float32)
        self.rail.set_positions(self.rail_pos)
        self.cart.set_positions(self.cart_pos)
        self.pole.set_positions(self.pole_pos)

        self.rail_base_color = torch.ones((1, instances_per_scene, 4), dtype=torch.float32) * torch.tensor(self.rail_pos_color, dtype=torch.float32)
        start_col = torch.tensor(self.cart_pos_color_range[0], dtype=torch.float32)
        end_col = torch.tensor(self.cart_pos_color_range[1], dtype=torch.float32)
        t = torch.linspace(0.0, 1.0, steps=num_scenes, dtype=torch.float32)
        self.cart_base_color = (start_col.unsqueeze(0) + (end_col - start_col).unsqueeze(0) * t.unsqueeze(1))
        self.pole_base_color = torch.ones((num_scenes, instances_per_scene, 4), dtype=torch.float32) * torch.tensor(self.pole_pos_color, dtype=torch.float32)
        self.rail.set_colors(self.rail_base_color)
        self.cart.set_colors(self.cart_base_color)
        self.pole.set_colors(self.pole_base_color)

        self.pole_hpr = torch.zeros((num_scenes, instances_per_scene, 3), dtype=torch.float32)
        self.pole_hpr[:, :, 1] = math.pi * 0.5
        self.pole.set_hprs(self.pole_hpr)

        # Buffers mapped from state
        self.cart_x_pos = self.cart_pos[:, :, 0:1].clone()
        self.pole_theta = torch.zeros_like(self.cart_x_pos)

        self.add_camera()
        self._p3d_cam.set_positions(torch.tensor([5, 5, 2], dtype=torch.float32))
        self._p3d_cam.look_at(torch.tensor([0, 0, 0], dtype=torch.float32))
        self.add_light()
        self.setup_environment()

    def _step(self, state_batch: torch.Tensor | None = None):
        # Accept direct state [B, 4] where x and theta are at indices 0 and 2
        if state_batch is None:
            return
        state_t = torch.as_tensor(state_batch, dtype=torch.float32).detach().cpu()
        B = int(state_t.shape[0])
        # Truncate/expand to num_scenes
        N = int(self.cfg.num_scenes)
        if B != N:
            if B < N:
                pad = state_t[-1:].repeat(N - B, 1)
                state_t = torch.cat([state_t, pad], dim=0)
            else:
                state_t = state_t[:N]
        x = state_t[:, 0:1]
        theta = state_t[:, 2:3]
        self.cart_x_pos[:, :, 0] = x
        self.cart_pos[:, :, 0:1] = self.cart_x_pos
        self.cart.set_positions(self.cart_pos)
        self.pole_pos[:, :, 0:1] = self.cart_x_pos
        self.pole.set_positions(self.pole_pos, lazy=True)
        self.pole_theta[:, :, 0] = theta
        self.pole_hpr[:, :, 1:2] = self.pole_theta
        self.pole.set_hprs(self.pole_hpr)

class CartPoleEnv(P3DEnv):
    def __init__(self, 
                renderer: CartPoleRenderer,
                cfg: CartPoleConfig | dict | None = None,
                **cfg_overrides,
                ):

        # if cfg is None:
        #     cfg = CartPoleConfig()
        # cfg = CartPoleConfig.from_config(cfg, **cfg_overrides)

        super().__init__(renderer=renderer, cfg=cfg, device=torch.device(cfg.device), batch_size=torch.Size([cfg.num_scenes]))

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

        self.seed = int(cfg.seed if cfg is not None else 0)
        # Keep render spec consistent across workers; gate heavy rendering at runtime
        self.render = bool(cfg.render if cfg is not None else True)

        # Convert degree-based config ranges to radians for internal use
        th0_lo, th0_hi = float(cfg.init_theta_range_deg[0]), float(cfg.init_theta_range_deg[1])
        thd_lo, thd_hi = float(cfg.init_theta_dot_range_deg[0]), float(cfg.init_theta_dot_range_deg[1])
        self._init_theta_range_rad = (
            math.radians(min(th0_lo, th0_hi)),
            math.radians(max(th0_lo, th0_hi)),
        )
        self._init_theta_dot_range_rad = (
            math.radians(min(thd_lo, thd_hi)),
            math.radians(max(thd_lo, thd_hi)),
        )

        # Specs via base helper
        self.set_default_specs(
            direct_obs_dim=4,
            actions=2,
            with_pixels=self.render,
            pixels_only=False,
            discrete_actions=True,
        )

        if self.seed is not None:
            self.set_seed(self.seed)

    def _sample_initial_state(self, batch_shape: torch.Size) -> torch.Tensor:
        x = torch.empty(*batch_shape, 1, dtype=torch.float32, device=self.device).uniform_(*self.cfg.init_x_range)
        x_dot = torch.empty(*batch_shape, 1, dtype=torch.float32, device=self.device).uniform_(*self.cfg.init_x_dot_range)
        theta = torch.empty(*batch_shape, 1, dtype=torch.float32, device=self.device).uniform_(*self._init_theta_range_rad)
        theta_dot = torch.empty(*batch_shape, 1, dtype=torch.float32, device=self.device).uniform_(*self._init_theta_dot_range_rad)
        return torch.cat([x, x_dot, theta, theta_dot], dim=-1)

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

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(int(seed))

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        bs = self.batch_size if self.batch_size != torch.Size([]) else torch.Size([1])
        state = self._sample_initial_state(bs)
        step_count = torch.zeros(*bs, dtype=torch.long, device=self.device)
        done = torch.zeros(*bs, 1, dtype=torch.bool, device=self.device)
        td_fields = {
            "observation": state, # .to("cpu").contiguous(),
            "step_count": step_count, # .to("cpu").contiguous(),
            "done": done, # .to("cpu").contiguous(),
        }
        if self.render:
            try:
                pixels = self.render_pixels(state)
                td_fields["pixels"] = pixels #.to("cpu").contiguous().clone()
            except Exception:
                pass
        return TensorDict(td_fields, batch_size=self.batch_size)

    @torch.no_grad()
    def _step(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict.get("observation", None)

        # print(obs[:2][:5])
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
        # print(next_obs_raw[:2][:5], 'NEXT_OBS_RAW')
        done = self._termination(next_obs_raw, step_count)
        reward = self._reward(next_obs_raw, done)

        pixels = None
        if self.render:
            try:
                pixels = self.render_pixels(next_obs_raw)
                # print(pixels)
            except Exception:
                pixels = None

        if self.auto_reset:
            reset_state = self._sample_initial_state(next_obs_raw.shape[:-1])
            next_obs = torch.where(done, reset_state, next_obs_raw)
            next_step_count = torch.where(done.squeeze(-1), torch.zeros_like(step_count), step_count + 1)
        else:
            next_obs = next_obs_raw
            next_step_count = step_count + 1

        # # Move core fields to CPU for safe IPC
        # next_obs = next_obs.to("cpu").contiguous()
        # next_step_count = next_step_count.to("cpu").contiguous()
        # reward = reward.to("cpu").contiguous()
        # done = done.to("cpu").contiguous()
        # if pixels is not None:
        #     try:
        #         pixels = pixels.to("cpu").contiguous().clone()
        #     except Exception:
        #         pass

        # EnvBase.step expects _step to return the "next" data at root.
        # ParallelEnv will copy these keys into the shared "next" buffer.
        out_fields = {
            "observation": next_obs,
            "reward": reward,
            "done": done,
            "step_count": next_step_count,
        }

        if pixels is not None:
            out_fields["pixels"] = pixels #.clone()

        # # Debug prints
        # print(next_obs[:2][:5], 'NEXT_OBS HERE')
        # print(reward.sum().item(), 'REWARD')

        return TensorDict(out_fields, batch_size=self.batch_size)

# Quick test
if __name__ == "__main__":

    use_parallel_envs = False
    num_parallel_envs = 8
    num_scenes = 1024 #256
    tile_resolution = (64, 64)
    config = CartPoleConfig(num_scenes=num_scenes, tile_resolution=tile_resolution)
    if use_parallel_envs:
        num_workers = int(num_parallel_envs)
        env = CartPoleEnv.make_parallel_env(
            config=config,
            renderer_cls=CartPoleRenderer,
            num_workers=num_workers,
            mp_start_method="spawn",
            shared_memory=True,
        )

        td = env.reset()
        total = 0.0
        tm_start = time.time()
        steps = 200
        bs = env.batch_size if env.batch_size != torch.Size([]) else torch.Size([1])
        num_envs = 1

        print(f"batch size: {bs}")
        for d in bs:
            num_envs *= int(d)
        total_frames = steps * num_envs
        print(f"total frames: {total_frames}")

        for t in range(steps):
            a = env.action_spec.rand()
            td = td.clone()
            td["action"] = a
            td = env.step(td)
            total += td["next", "reward"].sum().item()
            # print(td['next', 'reward'].sum().item(), 'CURRENT_REWARD')
           #  print('td', td)
            # print(td['observation'][:2][:5], 'CURRENT')
            td = td["next"]
            # print(td['observation'][:2][:5], 'NEXT')
            # break

            # td = td["next"]
            # print(td['pixels'])
            # print(td['next']['pixels'])

            step_interval = int(getattr(config, 'save_every_steps', -1))
            if step_interval >= 0:
                do_save = (step_interval == 0) or (t % step_interval == 0)
                if do_save:
                    # print(td['observation'][:2][:5])
                    try:
                        pixels = td.get("pixels", None)
                        env.save_batch_examples(
                            indices=(
                                list(range(5)) + list(range(num_scenes-3,num_scenes+3)) +
                                list(range(num_scenes*num_parallel_envs-5,num_scenes*num_parallel_envs))
                                ),
                            pixels=pixels,
                            num=int(getattr(config, 'save_examples_num', 16)),
                            out_dir=getattr(config, 'save_out_dir', None),
                            filename_prefix="batch_example",

                        )
                    except Exception:
                        pass

        tm_end = time.time()
        fps = total_frames / (tm_end - tm_start)
        print(f"Time taken (ParallelEnv, workers={num_workers}): {tm_end - tm_start} seconds")
        print(f"FPS: {fps}")
        env.close()
    else:
        env = CartPoleEnv(renderer=CartPoleRenderer(config), cfg=config)
        td = env.reset()
        total = 0.0
        tm_start = time.time()
        steps = 500
        total_frames = steps * num_scenes

        for t in range(steps):
            a = env.action_spec.rand()
            td["action"] = a
            td = env.step(td)
            total += td["next", "reward"].sum().item()
            td = td["next"]
            # Periodic saving logic
            step_interval = int(getattr(config, 'save_every_steps', -1))
            if step_interval >= 0:
                do_save = (step_interval == 0) or (t % step_interval == 0)
                if do_save:
                    try:
                        pixels = td.get("pixels", None)
                        env.save_batch_examples(
                            indices=list(range(16)),
                            pixels=pixels,
                            num=int(getattr(config, 'save_examples_num', 16)),
                            out_dir=getattr(config, 'save_out_dir', None),
                            filename_prefix="batch_example",
                        )
                    except Exception:
                        pass
            # print(td)
            # break
        tm_end = time.time()
        fps = total_frames / (tm_end - tm_start)
        print(f"Time taken (single-process): {tm_end - tm_start} seconds")
        print(f"FPS: {fps}")