"""Compare ground-truth and predicted metric depth using a shared color scale."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(ground_truth, prediction, output):
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if ground_truth.ndim != 2 or prediction.ndim != 2:
        raise ValueError("Depth maps must both have shape (height, width).")
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Depth shapes differ: {ground_truth.shape} vs {prediction.shape}. "
            "Align the maps to the same image coordinates before comparing."
        )

    gt_valid = np.isfinite(ground_truth) & (ground_truth > 0)
    pred_valid = np.isfinite(prediction) & (prediction > 0)
    valid = gt_valid & pred_valid
    if not valid.any():
        raise ValueError("No pixels have valid positive depth in both maps.")

    difference = prediction[valid] - ground_truth[valid]
    mae = np.mean(np.abs(difference))
    rmse = np.sqrt(np.mean(difference ** 2))
    bias = np.mean(difference)
    error = np.full(ground_truth.shape, np.nan)
    error[valid] = np.abs(difference)

    # Share the actual depth range; never normalize the two maps independently.
    vmin = min(ground_truth[gt_valid].min(), prediction[pred_valid].min())
    vmax = max(ground_truth[gt_valid].max(), prediction[pred_valid].max())
    if vmin == vmax:
        vmax = vmin + 1
    depth_cmap = plt.get_cmap("viridis").copy()
    error_cmap = plt.get_cmap("magma").copy()
    depth_cmap.set_bad("#d9d9d9")
    error_cmap.set_bad("#d9d9d9")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.3), layout="constrained")
    for ax, values, mask, title in (
        (axes[0], ground_truth, gt_valid, "Ground truth"),
        (axes[1], prediction, pred_valid, "Prediction"),
    ):
        depth_image = ax.imshow(
            np.ma.array(values, mask=~mask), cmap=depth_cmap,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax.set_title(title)
    fig.colorbar(depth_image, ax=axes[:2], label="Depth (m)", shrink=0.8)

    error_image = axes[2].imshow(
        np.ma.masked_invalid(error), cmap=error_cmap, vmin=0,
        vmax=max(float(np.max(np.abs(difference))), 1e-12),
        interpolation="nearest",
    )
    axes[2].set_title("Absolute error |prediction - truth|")
    fig.colorbar(error_image, ax=axes[2], label="Absolute error (m)", shrink=0.8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"Depth comparison  |  MAE {mae:.3f} m  |  RMSE {rmse:.3f} m\n"
        f"Mean bias (prediction - truth): {bias:+.3f} m  |  "
        f"Evaluated pixels: {valid.sum():,} / {valid.size:,} ({valid.mean():.1%})\n"
        "Gray: missing / invalid depth; errors use pixels valid in both maps",
        fontsize=12,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"MAE: {mae:.6f} m; RMSE: {rmse:.6f} m; bias: {bias:+.6f} m")
    print(f"Valid comparison pixels: {valid.sum()} / {valid.size}")
    print(f"Saved figure: {output.resolve()}")
    return fig


def plot_near_comparison(ground_truth, prediction, image_path, output,
                         max_depth=20.0, error_threshold=3.0, top_k=100):
    """Plot the GT-defined near region and export the largest pixel errors."""
    gt = np.asarray(ground_truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if gt.ndim != 2 or pred.ndim != 2 or gt.shape != pred.shape:
        raise ValueError("Depth maps must have identical (height, width) shapes.")
    if not np.isfinite(max_depth) or max_depth <= 0:
        raise ValueError("--max-depth must be finite and positive.")
    if not np.isfinite(error_threshold) or error_threshold < 0 or top_k < 1:
        raise ValueError("--error-threshold must be finite and nonnegative; --top-k must be positive.")

    near = np.isfinite(gt) & (gt > 0) & (gt <= max_depth)
    valid = near & np.isfinite(pred) & (pred > 0)
    if not valid.any():
        raise ValueError("No valid depth pairs in the requested near region.")
    image = plt.imread(image_path)
    if image.shape[:2] != gt.shape:
        raise ValueError("The reference image and depth maps must have identical dimensions.")

    ys, xs = np.where(valid)
    delta = pred[valid] - gt[valid]
    errors = np.abs(delta)
    order = np.argsort(-errors, kind="stable")[:top_k]
    largest = order[0]
    max_x, max_y = int(xs[largest]), int(ys[largest])
    mae, rmse, bias = errors.mean(), np.sqrt(np.mean(delta ** 2)), delta.mean()
    high_error = np.zeros(gt.shape, dtype=bool)
    high_error[valid] = errors >= error_threshold
    error_map = np.full(gt.shape, np.nan)
    error_map[valid] = errors

    # Crop to the near-region bounding box, keeping original pixel coordinates.
    region_y, region_x = np.where(near)
    x0, x1 = int(region_x.min()), int(region_x.max()) + 1
    y0, y1 = int(region_y.min()), int(region_y.max()) + 1
    crop = np.s_[y0:y1, x0:x1]
    extent = (x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5)
    depth_cmap = plt.get_cmap("viridis").copy()
    error_cmap = plt.get_cmap("magma").copy()
    depth_cmap.set_bad("#d9d9d9")
    error_cmap.set_bad("#d9d9d9")
    vmin = min(gt[valid].min(), pred[valid].min())
    vmax = max(gt[valid].max(), pred[valid].max())
    if vmin == vmax:
        vmax = vmin + 1

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), layout="constrained")
    for ax, values, title in (
        (axes[0, 0], gt, "Ground truth (near pixels)"),
        (axes[0, 1], pred, "Prediction (same pixels)"),
    ):
        depth_image = ax.imshow(
            np.ma.array(values, mask=~valid)[crop], extent=extent,
            cmap=depth_cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax.set_title(title)
    fig.colorbar(depth_image, ax=list(axes[0]), label="Depth (m)", shrink=0.85)

    error_image = axes[1, 0].imshow(
        np.ma.masked_invalid(error_map[crop]), extent=extent, cmap=error_cmap,
        vmin=0, vmax=max(float(errors.max()), 1e-12), interpolation="nearest",
    )
    axes[1, 0].set_title("Absolute error (near pixels only)")
    fig.colorbar(error_image, ax=axes[1, 0], label="Absolute error (m)", shrink=0.85)
    axes[1, 1].imshow(image[crop], extent=extent, interpolation="nearest")
    overlay = np.zeros((*gt.shape, 4), dtype=np.float32)
    overlay[high_error] = (1, 0, 0, 0.5)
    axes[1, 1].imshow(overlay[crop], extent=extent, interpolation="nearest")
    axes[1, 1].set_title(
        f"Red: error >= {error_threshold:g} m "
        f"({high_error.sum():,} pixels, {high_error.sum() / valid.sum():.1%})"
    )
    for ax in axes.flat:
        ax.scatter([max_x], [max_y], marker="x", s=85, c="cyan", linewidths=2)
        ax.set_xlabel("x / column (0-based)")
        ax.set_ylabel("y / row (0-based)")
    axes[1, 0].annotate(
        f"Max @ ({max_x}, {max_y})\n"
        f"GT {gt[max_y, max_x]:.3f} m; pred {pred[max_y, max_x]:.3f} m\n"
        f"Absolute error {errors[largest]:.3f} m",
        xy=(max_x, max_y), xytext=(0.42, 0.80), textcoords="axes fraction",
        color="white", fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#222222", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="cyan"),
    )
    fig.suptitle(
        f"Near depth: 0 < ground truth <= {max_depth:g} m  |  "
        f"MAE {mae:.3f} m  |  RMSE {rmse:.3f} m  |  Bias {bias:+.3f} m\n"
        f"Evaluated: {valid.sum():,} pixels  |  Cyan x: maximum error  |  "
        "Gray: excluded / invalid depth",
        fontsize=12,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    csv_path = output.with_suffix(".top_pixels.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("rank", "x", "y", "ground_truth_m", "prediction_m",
                         "prediction_minus_truth_m", "absolute_error_m"))
        for rank, i in enumerate(order, 1):
            y, x = int(ys[i]), int(xs[i])
            writer.writerow((rank, x, y, gt[y, x], pred[y, x], delta[i], errors[i]))
    error_path = output.with_suffix(".error.npy")
    np.save(error_path, error_map.astype(np.float32))
    print(f"Near depth <= {max_depth:g} m; evaluated {valid.sum()} pixels")
    print(f"MAE: {mae:.6f} m; RMSE: {rmse:.6f} m; bias: {bias:+.6f} m")
    print(f"Error >= {error_threshold:g} m: {high_error.sum()} pixels")
    for rank, i in enumerate(order[:10], 1):
        y, x = int(ys[i]), int(xs[i])
        print(f"#{rank} (x={x}, y={y}): GT={gt[y, x]:.6f} m, "
              f"prediction={pred[y, x]:.6f} m, error={errors[i]:.6f} m")
    print(f"Saved figure: {output.resolve()}")
    print(f"Saved top {len(order)} pixels: {csv_path.resolve()}")
    print(f"Saved pixel error array (excluded pixels = NaN): {error_path.resolve()}")
    return fig


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=project_dir / "depth_z_pinhole_m.npy")
    parser.add_argument("--image", type=Path, default=project_dir / "image_pinhole.png")
    parser.add_argument("--model", type=Path, default=project_dir / "yolo26s-depth.pt")
    parser.add_argument("--prediction", type=Path, help="Read an existing depth .npy instead of running inference.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. mps or cpu.")
    parser.add_argument("--output", type=Path, default=project_dir / "depth_comparison.png")
    parser.add_argument("--max-depth", type=float, help="Compare only pixels with ground-truth depth <= this distance in meters.")
    parser.add_argument("--error-threshold", type=float, default=3.0, help="Highlight near pixels whose absolute error is at least this many meters.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of largest pixel errors to export for the near region.")
    parser.add_argument("--show", action="store_true", help="Also open the figure interactively.")
    args = parser.parse_args()

    ground_truth = np.load(args.ground_truth, allow_pickle=False)
    if args.prediction:
        prediction = np.load(args.prediction, allow_pickle=False)
    else:
        from ultralytics import YOLO

        result = YOLO(args.model).predict(
            source=str(args.image), imgsz=768, device=args.device, save=False,
        )[0]
        if result.depth is None:
            raise RuntimeError("The model returned no depth map.")
        prediction = result.depth.data.cpu().numpy()
        prediction_path = args.output.with_suffix(".prediction.npy")
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(prediction_path, prediction)
        print(f"Saved predicted depth: {prediction_path.resolve()}")

    if args.max_depth is None:
        fig = plot_comparison(ground_truth, prediction, args.output)
    else:
        fig = plot_near_comparison(
            ground_truth, prediction, args.image, args.output,
            args.max_depth, args.error_threshold, args.top_k,
        )
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
