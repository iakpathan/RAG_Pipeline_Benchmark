"""
visualizer.py — Comparison dashboard for all three pipelines
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = "results"

PIPELINE_COLORS = {
    "Standard RAG":    "#4C72B0",
    "HyDE RAG":        "#DD8452",
    "Multi-Query RAG": "#55A868",
}

METRIC_LABELS = {
    "faithfulness":       "Faithfulness\n(answer grounded in context)",
    "answer_relevancy":   "Answer Relevancy\n(answer addresses question)",
    "context_precision":  "Context Precision\n(retrieved docs are useful)",
}


def plot_results(all_scores: dict):
    """
    Generate a 3-panel comparison dashboard:
      Panel 1: Grouped bar chart — all metrics side by side
      Panel 2: Radar chart — per-pipeline shape
      Panel 3: Summary table
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pipeline_names = list(all_scores.keys())
    metric_keys    = list(METRIC_LABELS.keys())

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        "RAG Pipeline Benchmark: Standard vs HyDE vs Multi-Query",
        fontsize=16, fontweight="bold", y=1.02
    )

    # ── Panel 1: Grouped bar chart ────────────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    x      = np.arange(len(metric_keys))
    width  = 0.25
    offset = -(len(pipeline_names) - 1) * width / 2

    for i, pipeline in enumerate(pipeline_names):
        scores = [all_scores[pipeline].get(m, 0) for m in metric_keys]
        bars   = ax1.bar(
            x + offset + i * width,
            scores,
            width,
            label=pipeline,
            color=PIPELINE_COLORS[pipeline],
            alpha=0.85,
            edgecolor="white",
        )
        # Value labels on bars
        for bar, score in zip(bars, scores):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.2f}",
                ha="center", va="bottom", fontsize=8
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([METRIC_LABELS[m] for m in metric_keys], fontsize=8)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Score (0–1)")
    ax1.set_title("Metric Comparison")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: Radar chart ──────────────────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2, polar=True)
    num_metrics = len(metric_keys)
    angles      = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles     += angles[:1]   # close the polygon

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(
        [METRIC_LABELS[m].split("\n")[0] for m in metric_keys],
        fontsize=9
    )
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax2.grid(alpha=0.3)

    for pipeline in pipeline_names:
        values  = [all_scores[pipeline].get(m, 0) for m in metric_keys]
        values += values[:1]
        ax2.plot(
            angles, values,
            "o-", linewidth=2,
            color=PIPELINE_COLORS[pipeline],
            label=pipeline
        )
        ax2.fill(angles, values, alpha=0.1, color=PIPELINE_COLORS[pipeline])

    ax2.set_title("Radar Comparison", pad=20)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    # ── Panel 3: Summary table ────────────────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")

    col_labels = ["Pipeline"] + [METRIC_LABELS[m].split("\n")[0] for m in metric_keys]
    table_data = []
    for pipeline in pipeline_names:
        row = [pipeline] + [
            f"{all_scores[pipeline].get(m, 0):.3f}" for m in metric_keys
        ]
        table_data.append(row)

    # Highlight best value per metric column in green
    table = ax3.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2E5FA3")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Row color + best value highlight
    for i, pipeline in enumerate(pipeline_names):
        table[i + 1, 0].set_facecolor(PIPELINE_COLORS[pipeline])
        table[i + 1, 0].set_text_props(color="white", fontweight="bold")

    for j, metric in enumerate(metric_keys):
        best_pipeline = max(pipeline_names, key=lambda p: all_scores[p].get(metric, 0))
        best_row      = pipeline_names.index(best_pipeline) + 1
        table[best_row, j + 1].set_facecolor("#d4edda")
        table[best_row, j + 1].set_text_props(fontweight="bold")

    ax3.set_title("Scores (green = best per metric)", fontsize=10, pad=15)

    plt.tight_layout()
    out_path = f"{RESULTS_DIR}/comparison_dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Dashboard saved → {out_path}")


def print_score_table(all_scores: dict):
    """Print a clean text table to the terminal."""
    metric_keys = list(METRIC_LABELS.keys())
    header_width = 20
    col_width    = 16

    header = f"{'Pipeline':<{header_width}}" + "".join(
        f"{METRIC_LABELS[m].split(chr(10))[0]:>{col_width}}" for m in metric_keys
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for pipeline, scores in all_scores.items():
        row = f"{pipeline:<{header_width}}" + "".join(
            f"{scores.get(m, 0):>{col_width}.3f}" for m in metric_keys
        )
        print(row)
    print("=" * len(header))


if __name__ == "__main__":
    # Load saved scores and re-plot (useful for tweaking the chart)
    with open(f"{RESULTS_DIR}/scores.json") as f:
        scores = json.load(f)
    plot_results(scores)
    print_score_table(scores)
