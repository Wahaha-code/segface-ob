# SegFace Real-Time Setup

## Environment

```bash
conda activate segface
```

The environment was created with Python 3.10 and includes PyTorch CUDA, OpenCV,
HuggingFace Hub, and Orbbec SDK v2 Python bindings (`pyorbbecsdk2`).

## Weights

The Swin-Base LaPa 512 checkpoint is here:

```text
weights/swinb_lapa_512/model_299.pt
```

It corresponds to:

```text
dataset=lapa
backbone=segface_lapa
model=swin_base
input_resolution=512
```

## Run With Orbbec Gemini 335L

```bash
conda activate segface
python realtime_segmentation.py --source orbbec
```

If Linux cannot open the camera because of USB permissions, install the Orbbec
udev rules once:

```bash
python -c "import pyorbbecsdk, os; print(os.path.dirname(pyorbbecsdk.__file__))"
sudo python /home/wenwu/miniconda3/envs/segface/lib/python3.10/site-packages/pyorbbecsdk/shared/setup_env.py
```

Unplug and reconnect the camera after applying udev rules.

## OpenCV Fallback

For quick model/camera testing through a standard RGB device:

```bash
conda activate segface
python realtime_segmentation.py --source opencv --camera-index 0
```

Press `q` or `Esc` to quit.
