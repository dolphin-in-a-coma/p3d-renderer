from abc import ABC, abstractmethod
import math
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase, ParallelEnv
from torchrl.data.tensor_specs import Composite, Unbounded, Categorical

# Optional renderer / config (kept optional to avoid hard deps when not rendering)
try:
    from .config import PBRConfig
    from .renderer.renderer import PBRRenderer
except ImportError:
    # Fallback when run as a script (no package parent)
    from config import PBRConfig
    from renderer.renderer import PBRRenderer

class PBREnv(EnvBase, ABC):

    def __init__(
        self,
        renderer: PBRRenderer,
        cfg: PBRConfig | None = None,
        device: str | torch.device = "cpu",
        batch_size: torch.Size = torch.Size([128]),
        **kwargs,
    ) -> None:
        super().__init__(device=torch.device(device), batch_size=batch_size, **kwargs)
        self._renderer = renderer
        self.cfg = cfg if cfg is not None else renderer.cfg
        
    # Minimal helpers to configure specs
    def set_default_specs(
        self,
        *,
        direct_obs_dim: int | None = None,
        actions: int | None = None,
        with_pixels: bool = False,
        pixels_only: bool = False,
        discrete_actions: bool = True,
    ) -> None:
        assert with_pixels or not pixels_only, "Cannot set both with_pixels and pixels_only to True."

        bs = self.batch_size if self.batch_size != torch.Size([]) else torch.Size([1])
        fields: dict[str, Unbounded] = {}

        # Pixel observation
        if with_pixels:
            C = int(self.cfg.num_channels)
            W = int(self.cfg.tile_resolution[0])
            H = int(self.cfg.tile_resolution[1])
            fields["pixels"] = Unbounded(
                shape=bs + torch.Size([C, H, W]), dtype=torch.uint8, device=self.device
            )

        # Direct observation
        if not pixels_only:
            if direct_obs_dim is None:
                raise ValueError("direct_obs_dim must be provided when pixels_only=False.")
            fields["observation"] = Unbounded(
                shape=bs + torch.Size([int(direct_obs_dim)]), dtype=torch.float32, device=self.device
            )
        self.observation_spec = Composite(**fields, shape=bs)

        # Actions
        if actions is None:
            raise ValueError("actions must be provided.")
        if discrete_actions:
            self.action_spec = Categorical(
                n=int(actions), shape=bs, dtype=torch.long, device=self.device
            )
        else:
            self.action_spec = Unbounded(
                shape=bs + torch.Size([int(actions)]), dtype=torch.float32, device=self.device
            )
        # Reward / Done
        self.reward_spec = Unbounded(
            shape=bs + torch.Size([1]), dtype=torch.float32, device=self.device
        )
        self.done_spec = Unbounded(
            shape=bs + torch.Size([1]), dtype=torch.bool, device=self.device
        )

    @abstractmethod
    def _step(self, tensordict: TensorDict) -> TensorDict:
        pass

    @abstractmethod
    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        pass

    def render_pixels(self, obs: torch.Tensor | None = None) -> torch.Tensor:
        if self._renderer is None:
            raise RuntimeError("Renderer is not initialized. Construct env with a renderer and pass it to PBREnv.")
        return self._renderer.step(obs)

    def save_batch_examples(
        self,
        *,
        pixels: torch.Tensor | None = None,
        obs: torch.Tensor | None = None,
        indices: list[int] | None = None,
        num: int = 16,
        out_dir: str | None = None,
        filename_prefix: str = "batch_example",
    ) -> str:

        if self.cfg.num_workers > 1 and self.cfg.worker_index != 0:
            return
        """
        Save a single image grid composed of N batch elements.

        - If `indices` is provided, selects those; otherwise picks `num` random batch elements.
        - If `pixels` is None, will render from `obs` via `render_pixels(obs)`.

        Returns the absolute path of the saved image.
        """
        import math
        import os
        import time
        import random
        from pathlib import Path

        # Acquire pixel batch [B, C, H, W]
        if pixels is None:
            if obs is None:
                pixels = self.render_pixels(None)
            else:
                pixels = self.render_pixels(obs)

        if pixels is None:
            raise RuntimeError("No pixels available to save.")
        
        if pixels.dim() == 5:
            pixels = torch.flatten(pixels, start_dim=0, end_dim=1)

        if pixels.dim() != 4:
            raise ValueError(f"Expected pixels shape [B,C,H,W], got {tuple(pixels.shape)}")

        B, C, H, W = pixels.shape
        k = min(max(1, int(num)), int(B))

        # Select indices
        if indices is not None and len(indices) > 0:
            sel = [int(i) for i in indices if 0 <= int(i) < B]
            if not sel:
                raise ValueError("Provided indices are out of range.")
            if len(sel) > k:
                sel = sel[:k]
        else:
            perm = torch.randperm(B, device=pixels.device)[:k]
            sel = perm.tolist()

        imgs = pixels[sel]  # [k, C, H, W]

        # Move to CPU, ensure uint8, and convert to HWC
        imgs = imgs.detach().to("cpu")
        if imgs.dtype != torch.uint8:
            imgs = imgs.clamp(0, 255).to(torch.uint8)
        imgs = imgs.permute(0, 2, 3, 1).contiguous()  # [k, H, W, C]

        # Build grid canvas
        rows = int(math.ceil(math.sqrt(k)))
        cols = int(math.ceil(k / rows))
        canvas = torch.zeros((rows * H, cols * W, C), dtype=torch.uint8)
        for n in range(k):
            r = n // cols
            c = n % cols
            y0, y1 = r * H, (r + 1) * H
            x0, x1 = c * W, (c + 1) * W
            canvas[y0:y1, x0:x1] = imgs[n]

        # Output path and filename
        ts = time.strftime("%Y%m%d_%H%M%S")
        rand_code = f"{random.randint(0, 999999):06d}"
        ext = "png"  # using .png by default
        out_path = Path(out_dir) if out_dir is not None else Path(".")
        out_path.mkdir(parents=True, exist_ok=True)
        fname = f"{filename_prefix}_{ts}_{rand_code}.{ext}"
        fpath = (out_path / fname).resolve()

        # Save via PIL if available, fallback to imageio or matplotlib
        try:
            from PIL import Image  # type: ignore
            Image.fromarray(canvas.numpy()).save(str(fpath))
        except Exception:
            try:
                import imageio.v2 as imageio  # type: ignore
                imageio.imwrite(str(fpath), canvas.numpy())
            except Exception:
                try:
                    import matplotlib.pyplot as plt  # type: ignore
                    plt.imsave(str(fpath), canvas.numpy())
                except Exception as e:
                    raise RuntimeError(f"Failed to save image: {e}")

        return str(fpath)

    @classmethod
    def make_parallel_env(
        cls,
        *,
        config: "PBRConfig",
        renderer_cls: type["PBRRenderer"],
        num_workers: int,
        mp_start_method: str = "spawn",
        shared_memory: bool = True,
    ) -> ParallelEnv:
        """
        Generic factory to create a TorchRL ParallelEnv for any PBREnv subclass.
        """
        import multiprocessing as mp

        try:
            mp.set_start_method(mp_start_method, force=True)
        except RuntimeError:
            # start method already set in this process
            pass

        create_env_kwargs = [
            {
                "env_cls": cls,
                "renderer_cls": renderer_cls,
                "config": config,
                "worker_index": i,
                "num_workers": num_workers,
            }
            for i in range(int(num_workers))
        ]

        return ParallelEnv(
            int(num_workers),
            create_env_fn=cls._make_env_worker,
            shared_memory=shared_memory,
            mp_start_method=mp_start_method,
            create_env_kwargs=create_env_kwargs,
        )

    @staticmethod
    def _make_env_worker(
        env_cls: type["PBREnv"],
        renderer_cls: type["PBRRenderer"],
        config: "PBRConfig",
        worker_index: int,
        num_workers: int,
    ) -> "PBREnv":
        """
        Spawn-safe worker factory to build a PBREnv subclass instance with its renderer.
        """
        cfg = PBRConfig.from_config(config, worker_index=worker_index, num_workers=num_workers)
        cfg.worker_index = worker_index
        try:
            if getattr(cfg, "seed", None) is not None:
                cfg.seed = int(cfg.seed) + int(worker_index)
        except Exception:
            # If seed is not present or not an int, ignore
            pass

        renderer = renderer_cls(cfg)
        env = env_cls(renderer=renderer, cfg=cfg)
        return env

    # # ---- Rendering helpers ----
    # def _num_instances(self) -> int:
    #     if self.batch_size == torch.Size([]):
    #         return 1
    #     n = 1
    #     for d in self.batch_size:
    #         n *= int(d)
    #     return max(1, n)

    # # Base class assumes renderer is constructed and passed in.
    # def sync_renderer_from_observation(self, obs: torch.Tensor) -> None:
    #     """
    #     Optional hook to be overridden by subclasses to reflect the current observation
    #     into the renderer's scene graph (e.g., set node transforms/colors per instance).
    #     Default is a no-op.
    #     """
    #     return None

    # def render_pixels(self, sync_from_obs: bool = True) -> torch.Tensor:
    #     if self._renderer is None:
    #         raise RuntimeError("Renderer is not initialized. Construct env with a renderer and pass it to PBREnv.")
    #     # if sync_from_obs:
    #     #     try:
    #     #         obs = getattr(self, "_last_obs", None)
    #     #         if obs is not None:
    #     #             self.sync_renderer_from_observation(obs)
    #     #     except Exception:
    #     #         pass
    #     img = self._renderer.step()
    #     return img

    # # Utility to cache last observation
    # def __setattr__(self, key, value):
    #     super().__setattr__(key, value)
    #     if key == "_last_td" and isinstance(value, TensorDict):
    #         try:
    #             self._last_obs = value.get("observation", None)
    #         except Exception:
    #             self._last_obs = None


