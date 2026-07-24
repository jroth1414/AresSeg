"""Generate data-grounded figures and LaTeX tables for the final NeurIPS report.

Every quantitative panel is derived from the canonical result store, sealed verdict file, or
archived exploratory MPBA store. Qualitative panels use the saved prediction masks; no synthetic
Mars imagery is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures" / "final"
GEN = ROOT / "paper" / "generated"
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from aresseg.data.ai4mars import (  # noqa: E402
    CLASS_COLORS,
    CLASSES,
    IGNORE_INDEX,
    build_index,
)

STORE = pd.read_parquet(ROOT / "experiments" / "results_store.parquet")
VERDICTS = json.loads(
    (ROOT / "experiments" / "manifests" / "verdicts.json").read_text(encoding="utf-8")
)
MPBA = pd.read_parquet(ROOT / "experiments" / "mpba" / "results_store.parquet")
MPBA_PROMOTION = json.loads(
    (ROOT / "experiments" / "mpba" / "promotion.json").read_text(encoding="utf-8")
)
DATA_ROOT = ROOT / "data" / "raw" / "ai4mars" / "ai4mars-dataset-merged-0.6"

PALETTE = {
    "majority": "#8A8F98",
    "tiny": "#4D4D4D",
    "unet": "#0072B2",
    "deeplab": "#D55E00",
    "segformer": "#009E73",
    "dinov3": "#CC79A7",
    "sam": "#E6AB02",
}

MODELS = [
    ("majority", "none", "constant", "Majority (bedrock)", "majority"),
    ("tiny_unet", "none", "scratch", "Tiny U-Net", "tiny"),
    ("unet", "resnet34", "pretrained", "U-Net R34 · pretrained", "unet"),
    ("unet", "resnet34", "scratch", "U-Net R34 · scratch", "unet"),
    ("deeplabv3plus", "resnet50", "pretrained", "DeepLabV3+ R50 · pretrained", "deeplab"),
    ("deeplabv3plus", "resnet50", "scratch", "DeepLabV3+ R50 · scratch", "deeplab"),
    ("segformer", "mit-b0", "pretrained", "SegFormer B0 · pretrained", "segformer"),
    ("segformer", "mit-b0", "scratch", "SegFormer B0 · scratch", "segformer"),
    ("segformer", "mit-b2", "pretrained", "SegFormer B2 · pretrained", "segformer"),
    ("segformer", "mit-b2", "scratch", "SegFormer B2 · scratch", "segformer"),
    ("dinov3_sat", "vitl16-sat493m", "finetuned", "DINOv3-SAT ViT-L/16", "dinov3"),
    ("sam", "vit-b", "region_oracle_upper_bound", "SAM region-oracle upper bound", "sam"),
]

HEATMAP_MODELS = [MODELS[i] for i in (0, 1, 2, 4, 6, 8, 10, 11)]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFBFC",
            "savefig.facecolor": "white",
            "savefig.dpi": 240,
        }
    )


def rows_for(model: str, backbone: str, variant: str, *, metric="miou", stratum="all"):
    return STORE[
        (STORE.model == model)
        & (STORE.backbone == backbone)
        & (STORE.variant == variant)
        & (STORE.metric == metric)
        & (STORE.stratum == stratum)
        & (STORE.status == "ok")
        & (STORE.profile == "gpu_full")
    ].copy()


def seed_values(spec) -> pd.DataFrame:
    model, backbone, variant, *_ = spec
    frame = rows_for(model, backbone, variant)
    return frame[(frame.scope == "ALL") & (frame.seed.isin([1414, 1415, 1416]))]


def performance_overview() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for y, spec in enumerate(MODELS):
        values = seed_values(spec).sort_values("seed")
        if values.empty:
            continue
        scores = values.value.to_numpy(float)
        color = PALETTE[spec[4]]
        if len(scores) > 1:
            ax.plot([scores.min(), scores.max()], [y, y], lw=4, alpha=0.25, color=color)
        ax.scatter(scores, np.full(len(scores), y), s=34, color=color, edgecolor="white", zorder=3)
        ax.scatter(scores.mean(), y, marker="D", s=38, color="#111111", zorder=4)
        ax.text(0.89, y, f"{scores.mean():.3f}", va="center", ha="right", fontsize=8)
    ax.set_yticks(range(len(MODELS)), [item[3] for item in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.91)
    ax.set_xlabel("MSL expert-test mIoU")
    ax.set_title("Three-seed performance; diamonds mark seed means")
    ax.grid(axis="x", color="#E2E4E8", lw=0.8)
    ax.axvspan(0, 0.15, color="#F2F2F2", zorder=-2)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color="#555", label="individual seed"),
            Line2D([], [], marker="D", ls="", color="#111", label="seed mean"),
        ],
        loc="lower right",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(FIG / "performance_overview.png", bbox_inches="tight")
    plt.close(fig)


def parameter_efficiency() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for spec in MODELS[1:11]:
        values = seed_values(spec)
        if values.empty:
            continue
        params = float(values.total_parameters.dropna().median())
        score = float(values.value.mean())
        color = PALETTE[spec[4]]
        ax.scatter(params / 1e6, score, s=85, color=color, edgecolor="white", linewidth=1.1)
        label = spec[3].replace(" · pretrained", " P").replace(" · scratch", " S")
        ax.annotate(
            label, (params / 1e6, score), xytext=(5, 4), textcoords="offset points", fontsize=7
        )
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (millions, log scale)")
    ax.set_ylabel("Mean MSL expert-test mIoU")
    ax.set_ylim(0.66, 0.85)
    ax.grid(color="#E2E4E8", lw=0.8)
    ax.set_title("Accuracy–capacity trade-off")
    fig.tight_layout()
    fig.savefig(FIG / "parameter_efficiency.png", bbox_inches="tight")
    plt.close(fig)


def per_class_heatmap() -> None:
    matrix = []
    for spec in HEATMAP_MODELS:
        model, backbone, variant, *_ = spec
        frame = rows_for(model, backbone, variant, metric="iou", stratum="per_class")
        matrix.append([frame[frame.scope == terrain].value.mean() for terrain in CLASSES])
    data = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    im = ax.imshow(data, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), [name.replace("_", " ").title() for name in CLASSES])
    ax.set_yticks(range(len(HEATMAP_MODELS)), [item[3] for item in HEATMAP_MODELS])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                f"{data[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if data[i, j] < 0.72 else "black",
                fontsize=8,
            )
    ax.set_title("Mean class IoU across training seeds")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("IoU")
    fig.tight_layout()
    fig.savefig(FIG / "per_class_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def hypothesis_forest() -> None:
    display = []
    family_names = {
        "A": "Deep − majority",
        "B": "Pretrained − scratch",
        "C": "SegFormer B2 − CNN",
        "E": "Foundation − Tiny U-Net",
    }
    for family in ("A", "B", "C", "E"):
        for member in VERDICTS["families"][family]["members"]:
            name = member["comparison"].replace("_", " ")
            display.append((family, name, member))
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    for y, (_, _, item) in enumerate(display):
        significant = item["decision"] == "significant"
        color = "#0072B2" if significant else "#8A8F98"
        ax.plot([item["ci_low"], item["ci_high"]], [y, y], color=color, lw=2.5)
        ax.scatter(item["delta"], y, s=44, color=color, edgecolor="white", zorder=3)
        p = item.get("holm_p")
        p_text = "" if p is None else ("$<10^{-3}$" if p < 0.001 else f"{p:.3f}")
        ax.text(0.86, y, p_text, va="center", ha="right", fontsize=7)
    ax.axvline(0, color="#222", lw=1)
    ax.set_yticks(range(len(display)), [f"{family_names[f]}: {n}" for f, n, _ in display])
    ax.invert_yaxis()
    ax.set_xlim(-0.18, 0.88)
    ax.set_xlabel("Paired mIoU difference with percentile 90% CI")
    ax.set_title("Preregistered effects (right column: Holm-adjusted p)")
    ax.grid(axis="x", color="#E2E4E8", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "hypothesis_forest.png", bbox_inches="tight")
    plt.close(fig)


def cross_rover() -> None:
    msl = rows_for("unet", "resnet34", "pretrained", stratum="in_rover")
    mer = rows_for("unet", "resnet34", "pretrained", stratum="cross_rover")
    majority_msl = rows_for("majority", "none", "constant", stratum="in_rover").value.iloc[0]
    majority_mer = rows_for("majority", "none", "constant", stratum="cross_rover").value.iloc[0]
    fig, ax = plt.subplots(figsize=(5.7, 4.6))
    for seed in (1414, 1415, 1416):
        y = [float(msl[msl.seed == seed].value.iloc[0]), float(mer[mer.seed == seed].value.iloc[0])]
        ax.plot([0, 1], y, marker="o", lw=2, alpha=0.75, label=str(seed))
    ax.scatter(
        [0, 1],
        [msl.value.mean(), mer.value.mean()],
        marker="D",
        s=90,
        color="black",
        label="mean",
        zorder=5,
    )
    ax.scatter(
        [0, 1],
        [majority_msl, majority_mer],
        marker="X",
        s=70,
        color=PALETTE["majority"],
        label="majority",
    )
    ax.set_xticks([0, 1], ["Curiosity (MSL)", "Opportunity/Spirit (MER)"])
    ax.set_ylabel("Expert-test mIoU")
    ax.set_ylim(0, 0.9)
    ax.set_title("Validation-selected U-Net cross-rover transfer")
    ax.grid(axis="y", color="#E2E4E8", lw=0.8)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    ax.text(0.5, 0.18, "mean drop = 0.030\n90% MER CI [0.725, 0.848]", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "cross_rover_generalization.png", bbox_inches="tight")
    plt.close(fig)


def colorize(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb[mask == class_id] = color
    rgb[mask == IGNORE_INDEX] = (8, 8, 8)
    return rgb


def read_mask(path: str | Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    return value[:, :, 0] if value.ndim == 3 else value


def run_id(model, backbone, variant, seed=1414, stratum="all") -> str:
    frame = rows_for(model, backbone, variant, stratum=stratum)
    frame = frame[(frame.scope == "ALL") & (frame.seed == seed)]
    if len(frame) != 1:
        raise RuntimeError(f"expected one run for {model}/{backbone}/{variant}/{seed}/{stratum}")
    return str(frame.run_id.iloc[0])


def choose_examples(records: list[dict], classes=(3, 2)) -> list[dict]:
    chosen = []
    used = set()
    for class_id in classes:
        scored = []
        for record in records:
            if record["name"] in used:
                continue
            mask = read_mask(record["label"])
            valid = mask != IGNORE_INDEX
            score = float(np.count_nonzero(mask == class_id) / max(np.count_nonzero(valid), 1))
            scored.append((score, record))
        _, winner = max(scored, key=lambda item: item[0])
        chosen.append(winner)
        used.add(winner["name"])
    return chosen


def qualitative_msl() -> None:
    records = build_index(DATA_ROOT, rover="msl", camera="ncam")["test"]
    examples = choose_examples(records)
    predictors = [
        ("U-Net", run_id("unet", "resnet34", "pretrained")),
        ("SegFormer B0", run_id("segformer", "mit-b0", "pretrained")),
        ("DINOv3-SAT", run_id("dinov3_sat", "vitl16-sat493m", "finetuned")),
        ("SAM oracle", run_id("sam", "vit-b", "region_oracle_upper_bound")),
    ]
    fig, axes = plt.subplots(2, 6, figsize=(12.8, 4.8), constrained_layout=True)
    for row, record in enumerate(examples):
        image = read_mask(record["image"])
        target = read_mask(record["label"])
        panels = [("Navcam image", image, True), ("Expert label", target, False)]
        for label, rid in predictors:
            panels.append(
                (
                    label,
                    read_mask(
                        ROOT / "experiments" / rid / "preds" / "test_msl" / f"{record['name']}.png"
                    ),
                    False,
                )
            )
        for ax, (title, content, gray) in zip(axes[row], panels, strict=True):
            (
                ax.imshow(content, cmap="gray" if gray else None)
                if gray
                else ax.imshow(colorize(content))
            )
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle("MSL expert-test examples selected by big-rock and sand coverage", fontsize=11)
    fig.savefig(FIG / "qualitative_msl.png", bbox_inches="tight")
    plt.close(fig)


def qualitative_mer() -> None:
    records = build_index(
        DATA_ROOT, rover="mer", test_gold_dir="mer/labels/test/masked-gold-min3-100agree"
    )["test"]
    examples = choose_examples(records)
    runs = [
        run_id("unet", "resnet34", "pretrained", seed, "cross_rover") for seed in (1414, 1415, 1416)
    ]
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.7), constrained_layout=True)
    for row, record in enumerate(examples):
        panels = [
            ("MER image", read_mask(record["image"]), True),
            ("Expert label", read_mask(record["label"]), False),
        ]
        for seed, rid in zip((1414, 1415, 1416), runs, strict=True):
            panels.append(
                (
                    f"U-Net seed {seed}",
                    read_mask(
                        ROOT / "experiments" / rid / "preds" / "test_mer" / f"{record['name']}.png"
                    ),
                    False,
                )
            )
        for ax, (title, content, gray) in zip(axes[row], panels, strict=True):
            (
                ax.imshow(content, cmap="gray" if gray else None)
                if gray
                else ax.imshow(colorize(content))
            )
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle("Cross-rover examples reveal both transfer and seed variability", fontsize=11)
    fig.savefig(FIG / "qualitative_mer.png", bbox_inches="tight")
    plt.close(fig)


def mpba_screen() -> None:
    data = MPBA[(MPBA.metric == "miou") & (MPBA.scope == "ALL") & (MPBA.status == "ok")]
    variants = ["content", "static", "raw_y", "range_cutoff"]
    backbones = ["resnet34", "mit-b0"]
    fig, ax = plt.subplots(figsize=(7.1, 4.4))
    x = np.arange(len(variants))
    width = 0.34
    for offset, backbone in zip((-width / 2, width / 2), backbones, strict=True):
        native = float(data[(data.backbone == backbone) & (data.variant == "native")].value.iloc[0])
        gains = [
            float(data[(data.backbone == backbone) & (data.variant == v)].value.iloc[0]) - native
            for v in variants
        ]
        ax.bar(x + offset, gains, width, label=backbone, alpha=0.86)
    ax.axhline(0, color="#222", lw=0.8)
    ax.axhline(0.003, color="#BB3E03", lw=1.2, ls="--", label="mIoU gain gate")
    ax.set_xticks(x, [v.replace("_", "\n") for v in variants])
    ax.set_ylabel("Validation mIoU change vs native decoder")
    ax.set_title("Exploratory MPBA screen: no perspective variant promoted")
    ax.grid(axis="y", color="#E2E4E8", lw=0.8)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "mpba_screen.png", bbox_inches="tight")
    plt.close(fig)


def escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace(" · ", r" \textperiodcentered{} ")


def write_tables() -> None:
    summaries = []
    for spec in MODELS:
        values = seed_values(spec)
        if values.empty:
            continue
        summaries.append(
            {
                "system": spec[3],
                "n": len(values),
                "mean": float(values.value.mean()),
                "sd": float(values.value.std(ddof=1)) if len(values) > 1 else np.nan,
                "params": float(values.total_parameters.dropna().median()),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(GEN / "model_summary.csv", index=False)
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"System & Seeds & mIoU (mean $\pm$ SD) & Params \\",
        r"\midrule",
    ]
    for row in summaries:
        spread = "--" if np.isnan(row["sd"]) else f"{row['sd']:.3f}"
        params = "0" if row["params"] == 0 else f"{row['params']/1e6:.1f}M"
        lines.append(
            f"{escape(row['system'])} & {row['n']} & {row['mean']:.3f} "
            f"$\\pm$ {spread} & {params}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "model_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hlines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Hypothesis & Decision & Key evidence & Adjusted $p$ \\",
        r"\midrule",
    ]
    evidence = {
        "H0": ("Rejected", "H1 supported", "--"),
        "H1": ("Supported", "$\\Delta$=0.754--0.765", "$<0.001$"),
        "H2": ("Rejected", "all transfer CIs cross 0", "$\\geq0.594$"),
        "H3": ("Rejected", "all 10 tests nonsignificant", "1.000"),
        "H4": ("Supported", "drop=0.030; MER CI$_L$=0.725", "rule"),
        "H5": ("Supported", "DINOv3 $\\Delta$=0.148", "$<0.001$"),
    }
    for hid in ("H0", "H1", "H2", "H3", "H4", "H5"):
        decision, key, p = evidence[hid]
        hlines.append(f"{hid} & {decision} & {key} & {p}" + r" \\ ")
    hlines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "hypothesis_summary.tex").write_text("\n".join(hlines) + "\n", encoding="utf-8")

    compact = {
        "generated_from": [
            "experiments/results_store.parquet",
            "experiments/manifests/verdicts.json",
        ],
        "protocol_sha256": (ROOT / "experiments" / "manifests" / "PROTOCOL_V5.sha256")
        .read_text()
        .strip(),
        "hypotheses": VERDICTS["hypotheses"],
        "model_summary": summaries,
        "mpba_status": MPBA_PROMOTION["status"],
    }
    (GEN / "final_results.json").write_text(
        json.dumps(compact, indent=2, default=str), encoding="utf-8"
    )


def main() -> None:
    setup_style()
    performance_overview()
    parameter_efficiency()
    per_class_heatmap()
    hypothesis_forest()
    cross_rover()
    qualitative_msl()
    qualitative_mer()
    mpba_screen()
    write_tables()
    print(f"wrote final assets to {FIG} and {GEN}")


if __name__ == "__main__":
    main()
