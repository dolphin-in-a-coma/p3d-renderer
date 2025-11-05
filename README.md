### pybatchrender

Utilities to build fast-rendered 3D environments with Panda3D and TorchRL. Includes a lightweight renderer, camera/light helpers, and example environments/demos.

### Features
- Fast multi-scene rendering with a tiled offscreen buffer
- Simple helpers to add nodes, cameras, and lights
- Torch/TorchRL-friendly image tensors (uint8, BCHW)
- Optional image saving via Pillow/ImageIO/Matplotlib
- Shaders shipped with the package

### Requirements
- Python >= 3.10
- Dependencies (installed automatically): `panda3d`, `torch`, `tensordict`, `torchrl`
- Optional for saving images: `Pillow`, `imageio`, `matplotlib`

### Installation
- From PyPI (once published):

```bash
pip install -U pip
pip install pybatchrender
```

- From source (this repo):

```bash
pip install -U pip
pip install .
# or editable
pip install -e .
```

- With optional image-saving extras:

```bash
pip install "pybatchrender[images]"
```

- With optional CUDA interop extras (Linux/Windows with NVIDIA):

```bash
# CUDA 12.x systems (default)
pip install "pybatchrender[cuda]"

# If you are on CUDA 11.x, install the matching CuPy wheel first, e.g.:
pip install cupy-cuda11x
pip install "pybatchrender[cuda]"
```

Notes:
- CUDA extras are skipped on macOS automatically.
- GPU grabbing requires an NVIDIA GPU and driver with CUDA, plus OpenGL.

### CUDA prerequisites (Ubuntu)
Detect your NVIDIA driver version, then install matching GL libs:

```bash
sudo apt-get update
DRV_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)
sudo apt-get install -y --no-install-recommends "libnvidia-gl-$DRV_MAJOR"
```

If needed, specify explicitly or fallback to basic EGL libs:

```bash
sudo apt-get install -y --no-install-recommends libnvidia-gl-{DRIVER_VERSION}
sudo apt-get install -y --no-install-recommends libegl1
```

Notes: Windows is not tested yet. Tested on macOS (Metal; no CUDA interop) and Ubuntu with NVIDIA.

### Quickstart
Render a small tiled batch offscreen and get a Torch tensor back as [B, C, H, W].

```python
from pybatchrender import PBRConfig
from pybatchrender import PBRRenderer

cfg = PBRConfig(num_scenes=4, tile_resolution=(64, 64), offscreen=True, report_fps=False)
renderer = PBRRenderer(cfg)

# Minimal setup
renderer.add_camera()
renderer.add_light()
renderer.add_node('models/box', instances_per_scene=1, shared_across_scenes=True)
renderer.setup_environment()

# Render one frame and get a BCHW tensor (uint8)
img_batch = renderer.step(return_pixels=True)
print(img_batch.shape)  # torch.Size([4, 3, 64, 64])
```

### Demos
The package ships a few demos you can run directly:

```bash
python -m pybatchrender.demo_cartpole
python -m pybatchrender.demo_cartpole_with_logic
python -m pybatchrender.demo_many_cubes
```

Notes:
- Demos use offscreen mode by default; they print FPS and can optionally save images.
- Demo models like `models/box` and `models/smiley` are standard Panda3D sample models.

### Using with TorchRL
`pybatchrender.env.PBREnv` provides a base class for TorchRL environments and utilities to define specs and render batches. See `pybatchrender/demo_cartpole_with_logic.py` for a full example environment with rendering.

### Package data
Shader sources under `pybatchrender/shaders/` are included automatically; no path configuration is needed.

### Development
- Build locally:

```bash
python -m build
```

- Run tests/demos from the repo root to validate rendering:

```bash
python -m pybatchrender.demo_cartpole
```

### License
MIT. See `LICENSE`.
