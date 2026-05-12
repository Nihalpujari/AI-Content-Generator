from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "runs" / "runs_log.csv"
OUT_DIR = BASE_DIR / "runs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "qwen/qwen3-32b",
    "gpt-4o",
    "llama-3.3-70b-versatile",
    "moonshotai/kimi-k2-instruct",
]

MODEL_LABELS = {
    "gpt-4o": "GPT-4o",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "moonshotai/kimi-k2-instruct": "Kimi K2",
    "qwen/qwen3-32b": "Qwen3 32B",
}

# Slightly brighter palette for better visibility in paper/report figures
PALETTE = {
    "gpt-4o": "#4F83D1",
    "llama-3.3-70b-versatile": "#F28E5C",
    "moonshotai/kimi-k2-instruct": "#59B36A",
    "qwen/qwen3-32b": "#D45A62",
}

NUM_COLS = [
    "optimized_score",
    "total_time_sec",
    "speed_wps",
    "score_improvement",
    "revision_rounds",
    "initial_score",
    "revised_score",
]

RUN_DATE = os.getenv("RUN_DATE", "2026-01-30")
FIG_DPI = 140
SAVE_DPI = 300

sns.set_theme(style="white", context="talk")
plt.rcParams.update(
    {
        "figure.dpi": FIG_DPI,
        "savefig.dpi": SAVE_DPI,
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
        "axes.grid": False,
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 0.95,
    }
)


POINT_LABEL_OFFSETS = {
    "qwen/qwen3-32b": (6, 6),
    "gpt-4o": (8, 6),
    # Move Llama label right/up so it is not clipped by the left edge
    "llama-3.3-70b-versatile": (0, -15),
    # Move Kimi slightly right/up to avoid overlap with Llama label
    "moonshotai/kimi-k2-instruct": (6, 6),
}


def style_axis(ax, *, ygrid=True, xgrid=False):
    ax.grid(False)
    if ygrid:
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    if xgrid:
        ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.15)


def save_fig(fig, filename: str):
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def add_value_labels(ax, fmt="{:.1f}", percent=False, fontweight="bold"):
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin if ymax > ymin else 1.0
    offset = yspan * 0.015

    for patch in ax.patches:
        h = patch.get_height()
        if pd.isna(h):
            continue
        label = f"{h:.0f}%" if percent else fmt.format(h)
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            h + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight=fontweight,
        )


def annotate_medians(ax, data_df, x_col, y_col, order, fmt="{:.1f}"):
    medians = data_df.groupby(x_col, observed=True)[y_col].median().reindex(order)
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin if ymax > ymin else 1.0
    for i, med in enumerate(medians):
        if pd.isna(med):
            continue
        ax.text(
            i,
            med + yspan * 0.02,
            fmt.format(med),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.9),
            zorder=7,
        )


def pareto_frontier(df_points: pd.DataFrame) -> pd.DataFrame:
    pareto = df_points.sort_values("avg_time_sec").copy()
    frontier = []
    best_score = -np.inf
    for _, row in pareto.iterrows():
        if row["avg_score"] > best_score:
            frontier.append(row)
            best_score = row["avg_score"]
    return pd.DataFrame(frontier)


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, str, float, float, int]:
    df = pd.read_csv(LOG_FILE)

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["date_utc"] = df["timestamp_dt"].dt.strftime("%Y-%m-%d")

    df = df[df["date_utc"] == RUN_DATE].copy()
    df = df[df["model_name"].isin(MODEL_ORDER)].copy()

    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["prompt_id", "model_name", "optimized_score", "total_time_sec"])
    df = df.sort_values("timestamp_dt").drop_duplicates(subset=["model_name", "prompt_id"], keep="last")

    counts = df.groupby("model_name")["prompt_id"].nunique().reindex(MODEL_ORDER)
    common_prompts = None
    for model in MODEL_ORDER:
        s = set(df.loc[df["model_name"] == model, "prompt_id"].unique())
        common_prompts = s if common_prompts is None else (common_prompts & s)

    common_prompts = common_prompts or set()
    common_n = len(common_prompts)

    print(f"\n=== Plotting Benchmark Results for date (UTC): {RUN_DATE} ===")
    print("Unique prompts per model:")
    for model in MODEL_ORDER:
        count = int(counts.loc[model]) if pd.notna(counts.loc[model]) else 0
        print(f"  {model:28s}: {count}")
    print(f"Common prompts across all 4 models: {common_n}")

    if common_n == 0:
        raise ValueError("No common prompts across all configured models for the selected RUN_DATE.")

    df = df[df["prompt_id"].isin(common_prompts)].copy()
    df["model_name"] = pd.Categorical(df["model_name"], categories=MODEL_ORDER, ordered=True)
    df["model_label"] = df["model_name"].map(MODEL_LABELS)

    agg = (
        df.groupby("model_name", as_index=False, observed=True)
        .agg(
            avg_time_sec=("total_time_sec", "mean"),
            avg_speed_wps=("speed_wps", "mean"),
            avg_score=("optimized_score", "mean"),
            median_score=("optimized_score", "median"),
            std_score=("optimized_score", "std"),
            rev_rate=("revision_rounds", lambda s: (s > 0).mean() * 100.0),
            avg_improvement=("score_improvement", lambda s: s[s > 0].mean() if (s > 0).any() else np.nan),
        )
        .sort_values("model_name")
        .reset_index(drop=True)
    )
    agg["model_label"] = agg["model_name"].map(MODEL_LABELS)

    baseline_row = agg.loc[agg["avg_score"].idxmin()]
    baseline_model = str(baseline_row["model_name"])
    baseline_score = float(baseline_row["avg_score"])
    baseline_time = float(baseline_row["avg_time_sec"])

    return df, agg, baseline_model, baseline_score, baseline_time, common_n

OPTIMIZED_THRESHOLD = 60

def plot_avg_score(agg: pd.DataFrame):
    plot_df = agg.sort_values("avg_score").copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    sns.barplot(
        data=plot_df,
        x="model_label",
        y="avg_score",
        order=plot_df["model_label"].tolist(),
        palette=[PALETTE[m] for m in plot_df["model_name"]],
        edgecolor="none",
        ax=ax,
    )

    ymax = max(100, plot_df["avg_score"].max() + 8)
    ax.set_ylim(0, ymax)
    add_value_labels(ax, fmt="{:.1f}")
    ax.set_title("Average Optimized Score by Model", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Optimized Score")
    ax.tick_params(axis="x", rotation=18)
    style_axis(ax, ygrid=True)
    
    # Add threshold line
    ax.axhline(
        OPTIMIZED_THRESHOLD,
        linestyle="--",
        linewidth=1.8,
        color="crimson",
        label=f"Threshold = {OPTIMIZED_THRESHOLD}"
    )

    # Add legend (top-right, compact like CLIP plot)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=10,
        handlelength=1.8,
        handletextpad=0.4,
        borderpad=0.25,
        labelspacing=0.2,
        borderaxespad=0.15,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
    )

    # Optional headroom so legend/value labels don’t feel cramped
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.1)

    save_fig(fig, "A1_avg_score_cloud_today.png")


def plot_time_vs_score(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    label_palette = {MODEL_LABELS[k]: PALETTE[k] for k in MODEL_ORDER}
    sns.scatterplot(
        data=df,
        x="total_time_sec",
        y="optimized_score",
        hue="model_label",
        style="model_label",
        palette=label_palette,
        s=78,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )

    # Overall linear trend across all points; label it explicitly in the legend.
    sns.regplot(
        data=df,
        x="total_time_sec",
        y="optimized_score",
        scatter=False,
        ci=None,
        line_kws={"linestyle": "--", "linewidth": 2.2, "alpha": 0.85, "color": "#444444", "label": "Trend (Quality vs Time)"},
        ax=ax,
    )

    ax.set_title("Time vs Optimized Score", pad=14)
    ax.set_xlabel("Total Time (sec)")
    ax.set_ylabel("Optimized Score")
    style_axis(ax, ygrid=True, xgrid=True)

    handles, labels = ax.get_legend_handles_labels()
    
    trend_line = Line2D(
        [0], [0],
        linestyle="--",
        linewidth=2.2,
        color="#444444",
        alpha=0.85,
        label="Trend (Quality vs Time)",
    )

    seen = set()
    final_h, final_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l != "model_label":
            final_h.append(h)
            final_l.append(l)
            seen.add(l)
    
    # Add trend line manually
    if "Trend (Quality vs Time)" not in seen:
        final_h.append(trend_line)
        final_l.append("Trend (Quality vs Time)")
 
    ax.legend(final_h, final_l, title="Model / Line", loc="lower right", fontsize=11, title_fontsize=11)
    save_fig(fig, "A2_time_vs_score_cloud_today.png")


def plot_speed_vs_score(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    sns.scatterplot(
        data=df,
        x="speed_wps",
        y="optimized_score",
        hue="model_label",
        style="model_label",
        palette={MODEL_LABELS[k]: PALETTE[k] for k in MODEL_ORDER},
        s=78,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )

    max_speed = float(df["speed_wps"].max()) if not df["speed_wps"].dropna().empty else 0.0
    tick_end = max(50, int(np.ceil(max_speed / 50.0) * 50))
    ticks = np.arange(50, tick_end + 1, 50)
    ax.set_xlim(0, tick_end + 10)
    ax.set_xticks(ticks)
    ax.set_title("Speed vs Optimized Score", pad=14)
    ax.set_xlabel("Speed (words/sec)")
    ax.set_ylabel("Optimized Score")
    style_axis(ax, ygrid=True, xgrid=True)
    ax.legend(title="Model", loc="upper right", fontsize=11, title_fontsize=11)
    save_fig(fig, "A3_speed_vs_score_cloud_today.png")


def plot_improvement(df: pd.DataFrame):
    revised = df[(df["revision_rounds"] > 0) & (df["score_improvement"] > 0)].copy()
    if revised.empty:
        return

    means = revised.groupby("model_name", observed=True, as_index=False)["score_improvement"].mean()
    means["model_label"] = means["model_name"].map(MODEL_LABELS)
    means = means.sort_values("score_improvement", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    sns.barplot(
        data=means,
        x="model_label",
        y="score_improvement",
        order=means["model_label"].tolist(),
        palette=[PALETTE[m] for m in means["model_name"]],
        edgecolor="none",
        ax=ax,
    )
    ax.set_ylim(0, means["score_improvement"].max() * 1.2)
    add_value_labels(ax, fmt="{:.1f}")
    ax.axhline(0, linestyle="--", linewidth=1.2, color="gray", alpha=0.6)
    ax.set_title("Average Score Improvement after Revision", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Score Improvement")
    ax.tick_params(axis="x", rotation=18)
    style_axis(ax, ygrid=True)
    save_fig(fig, "A4_avg_improvement_cloud_today.png")


def plot_revision_rate(agg: pd.DataFrame):
    plot_df = agg.sort_values("rev_rate", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    sns.barplot(
        data=plot_df,
        x="model_label",
        y="rev_rate",
        order=plot_df["model_label"].tolist(),
        palette=[PALETTE[m] for m in plot_df["model_name"]],
        edgecolor="none",
        ax=ax,
    )
    ax.set_ylim(0, 100)
    add_value_labels(ax, percent=True)
    ax.set_title("Revision Rate by Model", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Revision Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.tick_params(axis="x", rotation=18)
    style_axis(ax, ygrid=True)
    save_fig(fig, "A5_revision_rate_cloud_today.png")


def plot_score_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    order = [MODEL_LABELS[m] for m in MODEL_ORDER]
    sns.boxplot(
        data=df,
        x="model_label",
        y="optimized_score",
        order=order,
        palette=[PALETTE[m] for m in MODEL_ORDER],
        width=0.5,
        showfliers=False,
        linewidth=1.4,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="model_label",
        y="optimized_score",
        order=order,
        color="#222222",
        size=4.5,
        alpha=0.45,
        jitter=0.18,
        ax=ax,
    )
    annotate_medians(ax, df, "model_label", "optimized_score", order, fmt="{:.1f}")
    ax.set_title("Distribution of Optimized Scores", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Optimized Score")
    ax.tick_params(axis="x", rotation=18)
    style_axis(ax, ygrid=True)
    save_fig(fig, "A6_score_distribution_cloud_today.png")


def add_tradeoff_annotations(ax, agg: pd.DataFrame):
    for _, row in agg.iterrows():
        model = row["model_name"]
        label = row["model_label"]
        x = float(row["avg_time_sec"])
        y = float(row["avg_score"])
        dx, dy = POINT_LABEL_OFFSETS.get(model, (8, 7))
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy), ha="left", fontsize=11)


def tradeoff_legend_handles(baseline_model: str, include_pareto=False):
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[m], markeredgecolor="white", markersize=10, label=MODEL_LABELS[m])
        for m in MODEL_ORDER if m != baseline_model
    ]
    handles.append(
        Line2D([0], [0], marker="X", color="w", markerfacecolor=PALETTE[baseline_model], markeredgecolor="white", markersize=13, label=f"{MODEL_LABELS[baseline_model]} (baseline)")
    )
    if include_pareto:
        handles.append(Line2D([0], [0], linestyle="--", linewidth=2.2, color="#2F6DB3", label="Pareto frontier"))
    return handles


def plot_tradeoff(agg: pd.DataFrame, baseline_model: str, baseline_score: float, baseline_time: float):
    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    for _, row in agg.iterrows():
        model = row["model_name"]
        x = float(row["avg_time_sec"])
        y = float(row["avg_score"])
        is_baseline = model == baseline_model
        ax.scatter(
            x,
            y,
            s=220 if is_baseline else 150,
            marker="X" if is_baseline else "o",
            color=PALETTE[model],
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )

    add_tradeoff_annotations(ax, agg)

    ax.axhline(baseline_score, linestyle="--", linewidth=1.5, color="#2F6DB3", alpha=0.9)
    ax.axvline(baseline_time, linestyle="--", linewidth=1.5, color="#2F6DB3", alpha=0.9)

    ax.legend(
        handles=tradeoff_legend_handles(baseline_model),
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=11,
        title_fontsize=11,
    )
    xpad_left = max(1.4, agg["avg_time_sec"].max() * 0.06)
    xpad_right = max(1.6, agg["avg_time_sec"].max() * 0.16)
    ypad = max(0.35, (agg["avg_score"].max() - agg["avg_score"].min()) * 0.20)
    ax.set_xlim(max(0, agg["avg_time_sec"].min() - xpad_left), agg["avg_time_sec"].max() + xpad_right)
    ax.set_ylim(agg["avg_score"].min() - ypad, agg["avg_score"].max() + ypad)
    ax.set_title(f"Quality–Latency Tradeoff (Baseline: {MODEL_LABELS[baseline_model]})", pad=14)
    ax.set_xlabel("Average Total Time per Prompt (sec)")
    ax.set_ylabel("Average Optimized Score")
    style_axis(ax, ygrid=True, xgrid=True)
    save_fig(fig, "E1_tradeoff_time_vs_score_cloud_today.png")


def plot_pareto(agg: pd.DataFrame, baseline_model: str):
    frontier = pareto_frontier(agg[["model_name", "avg_time_sec", "avg_score"]].copy())
    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    for _, row in agg.iterrows():
        model = row["model_name"]
        x = float(row["avg_time_sec"])
        y = float(row["avg_score"])
        is_baseline = model == baseline_model
        ax.scatter(
            x,
            y,
            s=220 if is_baseline else 150,
            marker="X" if is_baseline else "o",
            color=PALETTE[model],
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )

    add_tradeoff_annotations(ax, agg)

    if not frontier.empty:
        ax.plot(
            frontier["avg_time_sec"],
            frontier["avg_score"],
            linestyle="--",
            linewidth=2.2,
            color="#2F6DB3",
            alpha=0.9,
            label="Pareto frontier",
            zorder=3,
        )

    ax.legend(
        handles=tradeoff_legend_handles(baseline_model, include_pareto=True),
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=11,
        title_fontsize=11,
    )
    xpad_left = max(1.6, agg["avg_time_sec"].max() * 0.06)
    xpad_right = max(1.8, agg["avg_time_sec"].max() * 0.16)
    ypad = max(0.35, (agg["avg_score"].max() - agg["avg_score"].min()) * 0.20)
    ax.set_xlim(max(0, agg["avg_time_sec"].min() - xpad_left), agg["avg_time_sec"].max() + xpad_right)
    ax.set_ylim(agg["avg_score"].min() - ypad, agg["avg_score"].max() + ypad)
    ax.set_title("Pareto Frontier: Time vs Optimized Score", pad=14)
    ax.set_xlabel("Average Total Time per Prompt (sec)")
    ax.set_ylabel("Average Optimized Score")
    style_axis(ax, ygrid=True, xgrid=True)
    save_fig(fig, "E2_pareto_frontier_cloud_today.png")


def export_summary_table(agg: pd.DataFrame, common_n: int):
    export_df = agg.copy()
    export_df["n_prompts"] = common_n
    export_df["model_label"] = export_df["model_name"].map(MODEL_LABELS)
    export_df = export_df[
        [
            "model_name",
            "model_label",
            "n_prompts",
            "avg_score",
            "median_score",
            "std_score",
            "avg_time_sec",
            "avg_speed_wps",
            "avg_improvement",
            "rev_rate",
        ]
    ].round(3)
    export_df.to_csv(OUT_DIR / "benchmark_summary_table.csv", index=False)


def main():
    df, agg, baseline_model, baseline_score, baseline_time, common_n = prepare_data()

    plot_avg_score(agg)
    plot_time_vs_score(df)
    plot_speed_vs_score(df)
    plot_improvement(df)
    plot_revision_rate(agg)
    plot_score_distribution(df)
    plot_tradeoff(agg, baseline_model, baseline_score, baseline_time)
    plot_pareto(agg, baseline_model)
    export_summary_table(agg, common_n)

    print(f"Success: Updated professional plots generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
