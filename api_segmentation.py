import argparse
import io
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from realtime_segmentation import (
    FACE_MERGE_IDS,
    LandmarkEstimator,
    load_model,
    overlay_mask,
    predict,
    preprocess,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SegFace face-mask HTTP API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-path",
        default="weights/swinb_lapa_512/model_299.pt",
        help="Path to the Swin-Base LaPa 512 checkpoint.",
    )
    parser.add_argument("--input-resolution", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def create_app(args):
    import torch

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = load_model(model_path, args.input_resolution, device)
    landmark_estimator = LandmarkEstimator()
    infer_lock = threading.Lock()

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "device": str(device),
                "input_resolution": args.input_resolution,
                "mask": "binary face mask",
            }
        )

    @app.post("/segment-face")
    def segment_face():
        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify({"error": "missing multipart field: image"}), 400

        image_bytes = uploaded.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if bgr_image is None:
            return jsonify({"error": "failed to decode image"}), 400

        with infer_lock:
            tensor, resized = preprocess(bgr_image, args.input_resolution, device)
            landmarks = landmark_estimator.estimate(resized)
            class_mask = predict(model, tensor, landmarks, args.input_resolution, device)

        face_mask = np.isin(class_mask, FACE_MERGE_IDS).astype(np.uint8) * 255
        face_mask = cv2.resize(
            face_mask,
            (bgr_image.shape[1], bgr_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        ok, encoded = cv2.imencode(".png", face_mask)
        if not ok:
            return jsonify({"error": "failed to encode mask"}), 500
        return Response(encoded.tobytes(), mimetype="image/png")

    @app.post("/segment-face-overlay")
    def segment_face_overlay():
        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify({"error": "missing multipart field: image"}), 400

        image_bytes = uploaded.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if bgr_image is None:
            return jsonify({"error": "failed to decode image"}), 400

        alpha = request.form.get("alpha", default=0.45, type=float)
        alpha = min(max(alpha, 0.0), 1.0)

        with infer_lock:
            tensor, resized = preprocess(bgr_image, args.input_resolution, device)
            landmarks = landmark_estimator.estimate(resized)
            class_mask = predict(model, tensor, landmarks, args.input_resolution, device)

        overlay = overlay_mask(bgr_image, class_mask, alpha, mode="face")
        ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return jsonify({"error": "failed to encode overlay"}), 500
        return Response(encoded.tobytes(), mimetype="image/jpeg")

    return app


def main():
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
