# SegFace Orbbec Face Mask

This repository packages SegFace for two practical use cases:

- Real-time face-region segmentation from an Orbbec Gemini 335L camera.
- HTTP image API that returns a binary face mask.

The current default output is a single face-region mask. The original LaPa model predicts 11 classes, but this project merges facial parts into one mask and hides background, inner mouth, and hair.

## Output Mask

Visible face mask classes:

```text
face_lr_rr, left brow, right brow, left eye, right eye, nose, upper lip, lower lip
```

Hidden classes:

```text
background, inner mouth, hair
```

For API output, the mask is a PNG:

```text
255 = face region
0   = non-face region
```

## Environment

Create or activate the project environment:

```bash
conda activate segface
```

The tested environment uses:

```text
Python 3.10
PyTorch 2.0.1 + CUDA 11.7
OpenCV
Flask
pyorbbecsdk2
```

If creating a fresh environment:

```bash
conda create -n segface python=3.10 -y
conda activate segface
pip install --extra-index-url https://download.pytorch.org/whl/cu117 \
  torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117
pip install numpy==1.23.4 opencv-python==4.8.1.78 scipy==1.8.1 \
  scikit-image==0.22.0 python-dotenv huggingface_hub tqdm Flask pyorbbecsdk2
```

## Model Weights

Weights are not committed to GitHub. Download the Swin-Base LaPa 512 checkpoint:

```bash
mkdir -p weights
python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="kartiknarayan/SegFace",
    filename="swinb_lapa_512/model_299.pt",
    local_dir="./weights",
)
PY
```

Expected path:

```text
weights/swinb_lapa_512/model_299.pt
```

## Real-Time Orbbec 335L

Connect the Orbbec Gemini 335L, then run:

```bash
conda activate segface
python realtime_segmentation.py --source orbbec
```

The default display is a single-color face mask overlay. Press `q` or `Esc` to quit.

To show the original 11 LaPa classes:

```bash
python realtime_segmentation.py --source orbbec --mask-mode classes
```

If Linux USB permissions block the camera:

```bash
sudo python /home/wenwu/miniconda3/envs/segface/lib/python3.10/site-packages/pyorbbecsdk/shared/setup_env.py
```

Unplug and reconnect the camera after installing udev rules.

## HTTP API

Start the API server:

```bash
conda activate segface
python api_segmentation.py --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Return a binary face mask PNG:

```bash
curl -X POST http://127.0.0.1:8000/segment-face \
  -F "image=@test.jpg" \
  --output mask.png
```

Return an overlay preview JPEG:

```bash
curl -X POST http://127.0.0.1:8000/segment-face-overlay \
  -F "image=@test.jpg" \
  -F "alpha=0.45" \
  --output overlay.jpg
```

## Files

```text
realtime_segmentation.py  Real-time Orbbec/OpenCV segmentation
api_segmentation.py       Flask HTTP API
REALTIME_SETUP.md         Local setup notes
network/                  SegFace model code
weights/                  Local model weights, ignored by git
```

## Notes

- The LaPa SegFace forward path requires 5 face landmarks. For real-time/API use, this project uses an OpenCV Haar face detector to estimate approximate 5-point landmarks.
- For production accuracy, replace `LandmarkEstimator` in `realtime_segmentation.py` with a stronger landmark detector.
- This repository is initialized with a fresh git history for the Orbbec/API workflow.

## Upstream

This project is based on SegFace:

```bibtex
@inproceedings{narayan2025segface,
  title={Segface: Face segmentation of long-tail classes},
  author={Narayan, Kartik and Vs, Vibashan and Patel, Vishal M},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={6},
  pages={6182--6190},
  year={2025}
}
```
