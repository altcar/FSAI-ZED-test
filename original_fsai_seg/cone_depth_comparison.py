"""Compare visible cones in this synthetic, planar-ground pinhole scene.

Cone masks come from ground-truth geometry, never from predicted depth.
The unobstructed ground below row 550 is used to fit inverse Z depth as
an affine function of image coordinates. Objects above this plane are cones
in this scene. Ground shadows are consequently excluded. This scene-specific
mask extraction is not a general-purpose cone segmentation model.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def extract_cones(gt, max_depth, ground_start_row=550,
                  plane_tolerance=1e-5, depth_gap=0.5):
    y, x = np.indices(gt.shape)
    valid = np.isfinite(gt) & (gt > 0)
    fit = valid & (y >= ground_start_row) & (y % 8 == 0) & (x % 8 == 0)
    if fit.sum() < 20:
        raise ValueError("Insufficient ground samples for this scene-specific fit.")
    design = np.column_stack((x[fit], y[fit], np.ones(fit.sum())))
    coefficients = np.linalg.lstsq(design, 1 / gt[fit], rcond=None)[0]
    inverse_ground = coefficients[0] * x + coefficients[1] * y + coefficients[2]
    residual = np.full(gt.shape, np.nan)
    residual[valid] = 1 / gt[valid] - inverse_ground[valid]
    ground_residual = float(np.max(np.abs(residual[valid & (y >= ground_start_row)])))
    if ground_residual >= plane_tolerance:
        raise ValueError("The assumed ground region does not fit a plane within tolerance.")

    # Extract whole visible objects before applying the distance cutoff.
    candidates = valid & (residual > plane_tolerance)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
    objects = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 4:
            continue
        component = labels == label
        depths = np.sort(gt[component])
        # Two overlapping yellow cones touch in the image, but have separate
        # GT depth ranges (about 5 m and 9 m). Split at gaps before relabeling.
        gaps = np.flatnonzero(np.diff(depths) > depth_gap)
        boundaries = (depths[gaps] + depths[gaps + 1]) / 2
        partitions = np.digitize(gt[component], boundaries)
        cy, cx = np.where(component)
        for partition in range(len(boundaries) + 1):
            part = np.zeros(gt.shape, dtype=np.uint8)
            selected = partitions == partition
            part[cy[selected], cx[selected]] = 1
            # Occlusion can leave a cone's body and a tiny piece of its base
            # disconnected. Preserve both pieces as one depth partition.
            mask = part.astype(bool)
            if mask.sum() >= 4 and np.median(gt[mask]) <= max_depth:
                objects.append(mask)
    objects.sort(key=lambda mask: float(np.median(gt[mask])))
    if not objects:
        raise ValueError("No cones found within the requested distance.")
    return objects, coefficients, ground_residual


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=root / "image_pinhole.png")
    parser.add_argument("--ground-truth", type=Path, default=root / "depth_z_pinhole_m.npy")
    parser.add_argument("--prediction", type=Path, default=root / "depth_comparison.prediction.npy")
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=root / "cone_depth_comparison.png")
    args = parser.parse_args()
    if not np.isfinite(args.max_depth) or args.max_depth <= 0:
        parser.error("--max-depth must be finite and positive.")
    gt = np.load(args.ground_truth, allow_pickle=False).astype(np.float64)
    pred = np.load(args.prediction, allow_pickle=False).astype(np.float64)
    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise ValueError(f"Could not read {args.image}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if gt.ndim != 2 or gt.shape != pred.shape or image.shape[:2] != gt.shape:
        raise ValueError("Image and depth maps must share the same height and width.")
    masks, plane, ground_residual = extract_cones(gt, args.max_depth)
    instance_map = np.zeros(gt.shape, dtype=np.uint16)
    error_map = np.full(gt.shape, np.nan, dtype=np.float32)
    rows = []
    for cone_id, mask in enumerate(masks, 1):
        instance_map[mask] = cone_id
        valid = mask & (gt <= args.max_depth) & np.isfinite(pred) & (pred > 0)
        if not valid.any():
            raise ValueError(f"Cone C{cone_id} has no valid paired depths.")
        delta = pred[valid] - gt[valid]
        errors = np.abs(delta)
        error_map[valid] = errors
        ys, xs = np.where(mask)
        eval_y, eval_x = np.where(valid)
        worst = int(np.argmax(errors))
        rows.append(dict(
            cone_id=f"C{cone_id}", pixels=int(valid.sum()),
            gt_mean_m=float(gt[valid].mean()), pred_mean_m=float(pred[valid].mean()),
            gt_median_m=float(np.median(gt[valid])), pred_median_m=float(np.median(pred[valid])),
            mae_m=float(errors.mean()), rmse_m=float(np.sqrt(np.mean(delta ** 2))),
            bias_m=float(delta.mean()), max_error_m=float(errors[worst]),
            max_error_x=int(eval_x[worst]), max_error_y=int(eval_y[worst]),
            center_x=float(xs.mean()), center_y=float(ys.mean()),
            x_min=int(xs.min()), y_min=int(ys.min()), x_max=int(xs.max()), y_max=int(ys.max()),
        ))

    evaluated = np.isfinite(error_map)
    all_delta = pred[evaluated] - gt[evaluated]
    overall_mae = float(np.abs(all_delta).mean())
    overall_rmse = float(np.sqrt(np.mean(all_delta ** 2)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.save(args.output.with_suffix(".instances.npy"), instance_map)
    np.save(args.output.with_suffix(".error.npy"), error_map)
    report = dict(
        max_depth_m=args.max_depth, cone_count=len(rows), evaluated_pixels=int(evaluated.sum()),
        pixel_mae_m=overall_mae, pixel_rmse_m=overall_rmse,
        extraction="GT plane residual > 1e-5 1/m; connected components >= 4 pixels; split depth gaps > 0.5 m",
        assumptions="This synthetic scene has planar ground and only cones above ground. Ground samples: rows >= 550.",
        selection="GT median <= cutoff; evaluate common valid cone pixels with GT <= cutoff.",
        exclusion="Ground, shadows, invalid depths, and coplanar contact boundary pixels are excluded.",
        plane_inverse_depth_coefficients=plane.tolist(), max_ground_inverse_depth_residual=ground_residual,
        objects=rows,
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(16, 10), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(1.2, 0.9, 2.2))
    image_ax = fig.add_subplot(grid[0, :])
    error_ax = fig.add_subplot(grid[1, :])
    depth_ax = fig.add_subplot(grid[2, 0])
    mae_ax = fig.add_subplot(grid[2, 1])
    image_ax.imshow(image)
    color_cycle = plt.get_cmap("tab20")
    spatial_order = np.argsort([row["center_x"] for row in rows])
    label_rows = {int(index): rank % 3 for rank, index in enumerate(spatial_order)}
    for i, (row, mask) in enumerate(zip(rows, masks)):
        color = color_cycle(i)
        image_ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.3)
        image_ax.annotate(
            row["cone_id"], xy=(row["center_x"], row["center_y"]),
            xytext=(row["center_x"], 505 + 19 * label_rows[i]),
            ha="center", va="top", fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#222222", alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=color, linewidth=1),
        )
    image_ax.set_title("Visible cones numbered by ground-truth median depth (nearest first)", loc="left")
    image_ax.set_ylim(565, 365)
    image_ax.set_xlim(210, gt.shape[1] - 1)
    image_ax.axis("off")

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#e6e6e6")
    norm = Normalize(vmin=0, vmax=float(np.nanmax(error_map)))
    error_image = error_ax.imshow(np.ma.masked_invalid(error_map), cmap=cmap, norm=norm, interpolation="nearest")
    error_ax.set_ylim(505, 365)
    error_ax.set_xlim(210, gt.shape[1] - 1)
    error_ax.set_title("Absolute error on cone pixels only (gray = excluded)", loc="left")
    error_ax.axis("off")
    fig.colorbar(error_image, ax=error_ax, label="Absolute error (m)", shrink=0.85, pad=0.01)

    positions = np.arange(len(rows))
    labels = [row["cone_id"] for row in rows]
    gt_means = [row["gt_mean_m"] for row in rows]
    pred_means = [row["pred_mean_m"] for row in rows]
    maes = [row["mae_m"] for row in rows]
    b1 = depth_ax.bar(positions - 0.2, gt_means, width=0.4, label="Ground truth", color="#268373")
    b2 = depth_ax.bar(positions + 0.2, pred_means, width=0.4, label="Prediction", color="#eb9952")
    depth_ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
    depth_ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)
    depth_ax.set_ylabel("Mean depth over the same cone pixels (m)")
    depth_ax.set_title("Per-cone depth comparison")
    depth_ax.set_ylim(0, max(max(gt_means), max(pred_means)) * 1.22)
    depth_ax.legend(loc="upper left", frameon=False)
    bars = mae_ax.bar(positions, maes, color=[cmap(norm(value)) for value in maes], width=0.65)
    mae_ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    mae_ax.set_ylabel("Mean pixel absolute error (m)")
    mae_ax.set_title("Per-cone error (lower is better)")
    mae_ax.set_ylim(0, max(maes) * 1.22)
    for ax in (depth_ax, mae_ax):
        ax.set_xticks(positions, labels)
        ax.set_xlabel("Cone ID (matches the image)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Cone-only depth comparison | ground truth <= {args.max_depth:g} m | {len(rows)} visible cones\n"
        f"{evaluated.sum():,} paired cone pixels | pixel MAE {overall_mae:.3f} m | RMSE {overall_rmse:.3f} m",
        fontsize=15,
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(f"Saved figure: {args.output.resolve()}")


if __name__ == "__main__":
    main()
