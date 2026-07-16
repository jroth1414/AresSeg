"""Generate reproducible midpoint-paper figures from local data and experiment artifacts.

The figures are descriptive only. They do not run the preregistered hypothesis tests and must not
be interpreted as confirmatory evidence until the required three-seed run set is complete.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from marsseg.data.ai4mars import CLASS_COLORS, CLASSES, IGNORE_INDEX, build_index  # noqa: E402

RESULTS = ROOT / "experiments" / "results_store.csv"
DATA_ROOT = ROOT / "data" / "raw" / "ai4mars" / "ai4mars-dataset-merged-0.6"

CONFIGS = [
    ("tiny_unet", "none", "scratch", "Tiny U-Net (scratch)", "#6c757d"),
    ("unet", "resnet34", "pretrained", "U-Net R34 (pretrained)", "#0072B2"),
    ("unet", "resnet34", "scratch", "U-Net R34 (scratch)", "#56B4E9"),
    ("deeplabv3plus", "resnet50", "pretrained", "DeepLabV3+ R50 (pretrained)", "#D55E00"),
    ("deeplabv3plus", "resnet50", "scratch", "DeepLabV3+ R50 (scratch)", "#E69F00"),
    ("segformer", "mit-b0", "pretrained", "SegFormer B0 (pretrained)", "#009E73"),
    ("segformer", "mit-b0", "scratch", "SegFormer B0 (scratch)", "#8BCB6B"),
    ("segformer", "mit-b2", "pretrained", "SegFormer B2 (pretrained)", "#6F4E7C"),
    ("segformer", "mit-b2", "scratch", "SegFormer B2 (scratch)", "#B497BD"),
    ("dinov3_sat", "vitl16-sat493m", "finetuned", "DINOv3-SAT ViT-L/16", "#CC79A7"),
]

RUNS_1414 = {
    "Tiny U-Net": "tiny_unet__scratch__none__in_rover__seed1414__gpu_full__b88a4624__gitc188b320__codebebd8b2d__20260710T213637673936Z",
    "U-Net": "unet__pretrained__resnet34__in_rover__seed1414__gpu_full__f8a3ff04__gitc188b320__codebebd8b2d__20260710T232715551236Z",
    "DeepLabV3+": "deeplabv3plus__pretrained__resnet50__in_rover__seed1414__gpu_full__71491561__gitc188b320__codebebd8b2d__20260711T052558603387Z",
    "SegFormer B0": "segformer__pretrained__mit-b0__in_rover__seed1414__gpu_full__22d89713__gitc188b320__codebebd8b2d__20260711T152946214899Z",
    "DINOv3-SAT": "dinov3_sat__finetuned__vitl16-sat493m__in_rover__seed1414__gpu_full__684c258f__gitc188b320__codebebd8b2d__20260713T043353711900Z",
}

COLORS_BY_LABEL = {
    "Tiny U-Net": "#6c757d",
    "U-Net": "#0072B2",
    "DeepLabV3+": "#D55E00",
    "SegFormer B0": "#009E73",
    "DINOv3-SAT": "#CC79A7",
}


def read_results() -> list[dict[str, str]]:
    with RESULTS.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def preliminary_miou(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if (
            row["metric"] == "miou"
            and row["stratum"] == "all"
            and row["status"] == "ok"
            and row["profile"] == "gpu_full"
        ):
            grouped[(row["model"], row["backbone"], row["variant"])].append(
                (int(row["seed"]), float(row["value"]))
            )

    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    y_positions = np.arange(len(CONFIGS))
    for y, (model, backbone, variant, _label, color) in enumerate(CONFIGS):
        values = sorted(grouped.get((model, backbone, variant), []))
        if not values:
            continue
        scores = np.asarray([value for _, value in values])
        if len(scores) > 1:
            ax.plot([scores.min(), scores.max()], [y, y], color=color, lw=2.4, alpha=0.55)
        for seed, score in values:
            marker = "o" if seed == 1414 else "s"
            ax.scatter(score, y, s=54, marker=marker, color=color, edgecolor="white", zorder=3)
        ax.scatter(scores.mean(), y, s=42, marker="D", color="black", zorder=4)
        ax.text(0.862, y, f"n={len(scores)}", va="center", ha="right", fontsize=8, color="#444444")

    ax.set_yticks(y_positions, [config[3] for config in CONFIGS])
    ax.invert_yaxis()
    ax.set_xlim(0.64, 0.87)
    ax.set_xlabel("Expert-test mean Intersection-over-Union")
    ax.set_title("Preliminary in-rover performance by available training seed")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#555555", label="seed 1414"),
        plt.Line2D([], [], marker="s", linestyle="", color="#555555", label="seed 1415"),
        plt.Line2D([], [], marker="D", linestyle="", color="black", label="available-seed mean"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)
    fig.text(
        0.01,
        0.005,
        "Descriptive midpoint results only; seed 1416 and preregistered paired inference are pending.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "preliminary_miou.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def training_curves() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for label, run_id in RUNS_1414.items():
        path = ROOT / "experiments" / "manifests" / run_id / "training_metrics.csv"
        points: dict[int, float] = {}
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("epoch") and row.get("val_miou"):
                    points[int(float(row["epoch"]))] = float(row["val_miou"])
        epochs = sorted(points)
        ax.plot(
            epochs,
            [points[epoch] for epoch in epochs],
            label=label,
            color=COLORS_BY_LABEL[label],
            linewidth=2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mIoU")
    ax.set_title("Representative seed-1414 validation learning curves")
    ax.set_ylim(0.55, 0.90)
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.legend(ncol=2, fontsize=8, frameon=True)
    fig.text(
        0.01,
        0.005,
        "Curves document pipeline convergence; final comparisons use canonical three-seed test artifacts.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(FIGURES / "training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def colorize(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb[mask == class_id] = color
    rgb[mask == IGNORE_INDEX] = (0, 0, 0)
    return rgb


def qualitative_predictions() -> None:
    name = "msl_ncam_nlb_556756308edr_f0651642ncam00285m1"
    record = next(rec for rec in build_index(DATA_ROOT, rover="msl")["test"] if rec["name"] == name)
    image = cv2.imread(record["image"], cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(record["label"], cv2.IMREAD_UNCHANGED)
    if image is None or target is None:
        raise FileNotFoundError("selected AI4Mars qualitative example is unavailable")

    panels: list[tuple[str, np.ndarray, bool]] = [
        ("Rover image", image, True),
        ("Expert label", target, False),
    ]
    for label, run_id in RUNS_1414.items():
        pred_path = ROOT / "experiments" / run_id / "preds" / "test_msl" / f"{name}.png"
        pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        if pred is None:
            raise FileNotFoundError(pred_path)
        panels.append((label, pred, False))

    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.6), constrained_layout=True)
    for ax, (title, content, grayscale) in zip(axes.flat, panels, strict=False):
        (
            ax.imshow(content, cmap="gray" if grayscale else None)
            if grayscale
            else ax.imshow(colorize(content))
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.set_title("Terrain classes", fontsize=9)
    legend_ax.legend(
        handles=[
            Patch(facecolor=np.asarray(CLASS_COLORS[i]) / 255, label=label.replace("_", " "))
            for i, label in enumerate(CLASSES)
        ]
        + [Patch(facecolor="black", label="ignore")],
        loc="center",
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Seed-1414 qualitative comparison on an expert-test scene with a large rock region",
        fontsize=11,
    )
    fig.savefig(FIGURES / "qualitative_predictions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = read_results()
    preliminary_miou(rows)
    training_curves()
    qualitative_predictions()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
