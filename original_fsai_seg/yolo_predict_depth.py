"""Save segmentation predictions and grayscale depth maps for each test image."""

import argparse
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=project_dir / "dataset" / "test" / "images",
        help="Input image or directory (searched recursively).",
    )
    parser.add_argument(
        "--output", type=Path, default=project_dir / "runs" / "segment_depth" / "predict",
        help="Directory for paired segmentation and depth images.",
    )
    parser.add_argument("--device", default=None, help="Inference device, e.g. mps, cpu or 0.")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        parser.error(f"Input does not exist: {source}")
    if source.is_file():
        image_paths = [source] if source.suffix.lower() in IMAGE_SUFFIXES else []
        source_dir = source.parent
    else:
        image_paths = sorted(
            p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        source_dir = source
    if not image_paths:
        print("No test images found.")
        return

    model = YOLO(project_dir / "yolov11_seg3" / "weights" / "yolov11_seg_test.pt")
    depth_model = YOLO(project_dir / "yolo26s-depth.pt")
    total_frames = 0
    total_inference_ms = 0.0
    total_elapsed = 0.0

    for image_path in image_paths:
        output_dir = args.output / image_path.relative_to(source_dir).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        # Include the original extension so foo.jpg and foo.png cannot overwrite each other.
        seg_path = output_dir / f"{image_path.name}_seg.png"
        depth_path = output_dir / f"{image_path.name}_depth.png"

        start = perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        result = model.predict(image, save=False, device=args.device)[0]
        depth_result = depth_model.predict(image, imgsz=768, save=False, device=args.device)[0]
        if depth_result.depth is None:
            raise RuntimeError(f"Depth model returned no depth map for {image_path}")

        # Match depth.py's grayscale display. These are per-image normalized
        # visualization values (0-255), not the original depth values in meters.
        depth = depth_result.depth.data.cpu().numpy()
        depth_image = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        for path, output_image in ((seg_path, result.plot()), (depth_path, depth_image)):
            if not cv2.imwrite(str(path), output_image):
                raise OSError(f"Could not save image: {path}")

        total_elapsed += perf_counter() - start
        inference_ms = result.speed["inference"] + depth_result.speed["inference"]
        total_frames += 1
        total_inference_ms += inference_ms
        if inference_ms > 0:
            print(
                f"{image_path.name}: segmentation + depth inference {inference_ms:.2f} ms, "
                f"FPS {1000.0 / inference_ms:.2f}"
            )
        print(f"Saved: {seg_path}\nSaved: {depth_path}")

    print(f"\nProcessed images: {total_frames}")
    if total_inference_ms > 0:
        print(f"Average combined model inference FPS: {total_frames * 1000.0 / total_inference_ms:.2f}")
    if total_elapsed > 0:
        print(f"Overall FPS (including loading, pre/post-processing and saving): {total_frames / total_elapsed:.2f}")
    print(f"Output directory: {args.output.resolve()}")


if __name__ == "__main__":
    main()
