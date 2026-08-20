import argparse
import time
from pathlib import Path

import cv2
import numpy as np


LABELS = [
    "background",
    "face_lr_rr",
    "lb",
    "rb",
    "le",
    "re",
    "nose",
    "ul",
    "im",
    "ll",
    "hair",
]

COLORS = np.array(
    [
        [0, 0, 0],
        [0, 153, 255],
        [102, 255, 153],
        [0, 204, 153],
        [255, 255, 102],
        [255, 255, 204],
        [255, 153, 0],
        [255, 102, 255],
        [102, 0, 51],
        [255, 204, 255],
        [255, 0, 102],
    ],
    dtype=np.uint8,
)

FACE_MERGE_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=np.uint8)
FACE_COLOR = np.array([0, 220, 255], dtype=np.uint8)


class OpenCVCamera:
    def __init__(self, index, width, height, fps):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open OpenCV camera index {index}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class OrbbecCamera:
    def __init__(self):
        try:
            from pyorbbecsdk import OBError, OBFormat, Pipeline
        except Exception as exc:
            raise RuntimeError(
                "pyorbbecsdk2 is not installed or cannot be imported"
            ) from exc

        self.OBError = OBError
        self.OBFormat = OBFormat
        self.pipeline = Pipeline()
        self.pipeline.start()

    def read(self):
        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None
        return self._frame_to_bgr(color_frame)

    def close(self):
        self.pipeline.stop()

    def _frame_to_bgr(self, frame):
        width = frame.get_width()
        height = frame.get_height()
        fmt = frame.get_format()
        data = np.asanyarray(frame.get_data())

        if fmt == self.OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if fmt == self.OBFormat.BGR:
            return np.resize(data, (height, width, 3))
        if fmt == self.OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == self.OBFormat.YUYV:
            image = np.resize(data, (height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        if fmt == self.OBFormat.UYVY:
            image = np.resize(data, (height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        if fmt == self.OBFormat.NV12:
            image = np.resize(data, (height * 3 // 2, width))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV12)
        if fmt == self.OBFormat.NV21:
            image = np.resize(data, (height * 3 // 2, width))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV21)
        raise RuntimeError(f"Unsupported Orbbec color format: {fmt}")


class LandmarkEstimator:
    def __init__(self):
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade)

    def estimate(self, bgr_image):
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return self._central_face(bgr_image.shape[1], bgr_image.shape[0])

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        return np.array(
            [
                [x + 0.32 * w, y + 0.38 * h],
                [x + 0.68 * w, y + 0.38 * h],
                [x + 0.50 * w, y + 0.55 * h],
                [x + 0.38 * w, y + 0.76 * h],
                [x + 0.62 * w, y + 0.76 * h],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _central_face(width, height):
        x = width * 0.5
        y = height * 0.5
        size = min(width, height) * 0.58
        return np.array(
            [
                [x - 0.18 * size, y - 0.18 * size],
                [x + 0.18 * size, y - 0.18 * size],
                [x, y],
                [x - 0.12 * size, y + 0.29 * size],
                [x + 0.12 * size, y + 0.29 * size],
            ],
            dtype=np.float32,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time SegFace LaPa segmentation for Orbbec Gemini 335L."
    )
    parser.add_argument(
        "--model-path",
        default="weights/swinb_lapa_512/model_299.pt",
        help="Path to the Swin-Base LaPa 512 checkpoint.",
    )
    parser.add_argument("--input-resolution", type=int, default=512)
    parser.add_argument("--source", choices=["orbbec", "opencv"], default="orbbec")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument(
        "--mask-mode",
        choices=["face", "classes"],
        default="face",
        help="face merges facial parts into one color; classes shows all 11 LaPa classes.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output", help="Optional video output path.")
    parser.add_argument("--debug", action="store_true", help="Print frame timing diagnostics.")
    return parser.parse_args()


def load_model(model_path, input_resolution, device):
    import torch

    from network import get_model

    model = get_model("segface_lapa", input_resolution, "swin_base").to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict_backbone", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def make_camera(args):
    if args.source == "orbbec":
        return OrbbecCamera()
    return OpenCVCamera(
        args.camera_index, args.camera_width, args.camera_height, args.camera_fps
    )


def preprocess(bgr_image, input_resolution, device):
    import torch

    resized = cv2.resize(bgr_image, (input_resolution, input_resolution))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device), resized


def predict(model, tensor, landmarks, input_resolution, device):
    import torch
    import torch.nn.functional as F

    labels = {"lnm_seg": torch.from_numpy(landmarks).unsqueeze(0).float().to(device)}
    dataset = torch.ones(1, dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(tensor, labels, dataset)
        output = F.interpolate(
            output,
            size=(input_resolution, input_resolution),
            mode="bilinear",
            align_corners=False,
        )
        return output.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)


def overlay_mask(frame, mask, alpha, mode):
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mode == "classes":
        color_mask = COLORS[mask]
        blended = cv2.addWeighted(frame, 1.0 - alpha, color_mask, alpha, 0)
        return np.where(mask[..., None] == 0, frame, blended)

    face_mask = np.isin(mask, FACE_MERGE_IDS)
    color_mask = np.zeros_like(frame)
    color_mask[face_mask] = FACE_COLOR
    blended = cv2.addWeighted(frame, 1.0 - alpha, color_mask, alpha, 0)
    return np.where(face_mask[..., None], blended, frame)


def draw_hud(frame, fps, source):
    cv2.putText(
        frame,
        f"{source} | {fps:4.1f} FPS | q/ESC quit",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    window_name = "SegFace Real-Time Segmentation"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.camera_width, args.camera_height)
        cv2.moveWindow(window_name, 80, 80)
        cv2.waitKey(1)
        print("Display window created.", flush=True)

    import torch

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    print(f"Using device: {device}", flush=True)
    print(f"Loading checkpoint: {model_path}", flush=True)
    model = load_model(model_path, args.input_resolution, device)
    print("Model loaded.", flush=True)
    landmarks = LandmarkEstimator()

    print(f"Opening camera source: {args.source}", flush=True)
    camera = make_camera(args)
    print("Camera opened.", flush=True)

    writer = None
    last = time.perf_counter()
    last_log = last
    fps = 0.0
    frames_seen = 0

    try:
        while True:
            t0 = time.perf_counter()
            frame = camera.read()
            if frame is None:
                if args.debug and time.perf_counter() - last_log > 2.0:
                    print("Waiting for color frame...", flush=True)
                    last_log = time.perf_counter()
                continue

            t1 = time.perf_counter()
            tensor, resized = preprocess(frame, args.input_resolution, device)
            lnd = landmarks.estimate(resized)
            t2 = time.perf_counter()
            mask = predict(model, tensor, lnd, args.input_resolution, device)
            t3 = time.perf_counter()
            vis = overlay_mask(frame, mask, args.alpha, args.mask_mode)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            draw_hud(vis, fps, args.source)
            frames_seen += 1

            if args.debug and now - last_log > 1.0:
                print(
                    "frame={} shape={} read={:.1f}ms prep={:.1f}ms infer={:.1f}ms fps={:.1f}".format(
                        frames_seen,
                        frame.shape,
                        (t1 - t0) * 1000,
                        (t2 - t1) * 1000,
                        (t3 - t2) * 1000,
                        fps,
                    ),
                    flush=True,
                )
                last_log = now

            if writer is None and args.output:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, 30, (vis.shape[1], vis.shape[0]))
            if writer is not None:
                writer.write(vis)

            if not args.no_display:
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        camera.close()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
