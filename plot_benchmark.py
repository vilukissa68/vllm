import json
import glob
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- PUBLICATION STYLE CONFIGURATION ---
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,  # Scaled down slightly for IEEE compliance
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "baseline": "#82c8f0",  # Blue
    "rans_unfused": "#FFDCA5",  # Orange
    "rans_fused": "#7DCDBE",  # Green
    "error": "#F5A5C8",  # Red
}

LABELS = {
    "baseline": "Baseline (BF16)",
    "rans_unfused": "rANS (Unfused)",
    "rans_fused": "rANS (Fused Kernel)",
}

# Standard IEEE Double Column Full Width is ~7.16 inches.
# Half-page height is roughly 3.5 to 4.5 inches.
FIG_WIDTH = 7.16
FIG_HEIGHT = 3.5


def load_data(filepath):
    print(f"Loading {filepath}...")
    with open(filepath, "r") as f:
        raw = json.load(f)

    if "results" in raw:
        data_list = raw["results"]
    elif "sweep_results" in raw and isinstance(raw["sweep_results"], list):
        data_list = raw["sweep_results"]
    else:
        print("Warning: Detected legacy JSON format. Some sweep plots may fail.")
        return None, raw["metadata"]

    if not data_list:
        raise ValueError("JSON file contains no result records.")

    df = pd.DataFrame(data_list)
    return df, raw["metadata"]


def plot_memory_wall(df, meta, output_dir, timestamp, target_batch_sizes=None):
    """
    PLOTS: Throughput (Y) vs VRAM Utilization (X).
    Consolidates subplots and annotates Offload/KV metrics directly on the points.
    """
    df_clean = df[df["avg_toks_sec"] > 0].copy()

    if target_batch_sizes:
        df_clean = df_clean[df_clean["batch_size"].isin(target_batch_sizes)]

    if df_clean.empty:
        print(f"No valid throughput data found for batch sizes: {target_batch_sizes}")
        return

    # Calculate optimal height to fit the 7.16 width requirement
    num_cols = len(df_clean["batch_size"].unique())
    aspect_ratio = (FIG_WIDTH / num_cols) / FIG_HEIGHT

    g = sns.relplot(
        data=df_clean,
        x="vram_util_config",
        y="avg_toks_sec",
        hue="mode",
        style="mode",
        col="batch_size",
        row="prompt_len",
        kind="line",
        palette=COLORS,
        markers=True,
        dashes=False,
        hue_order=["baseline", "rans_unfused", "rans_fused"],
        height=FIG_HEIGHT,
        aspect=aspect_ratio,
        facet_kws={"sharey": False},
    )

    def annotate_points(data, **kws):
        ax = plt.gca()
        agg_data = (
            data.groupby(["vram_util_config", "mode"])
            .agg(
                {
                    "avg_toks_sec": "mean",
                    "cpu_offload_gb": "max",
                    "kv_cache_tokens": "max",
                }
            )
            .reset_index()
        )

        for _, row in agg_data.iterrows():
            x = row["vram_util_config"]
            y = row["avg_toks_sec"]
            mode = row["mode"]
            offload = row["cpu_offload_gb"]
            kv = row["kv_cache_tokens"]

            kv_k = f"{int(kv)//1000}k" if kv > 0 else "0k"
            bbox_style = dict(
                boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75
            )

            if mode == "baseline":
                text = (
                    f"Off: {offload:.1f}G\nKV: {kv_k}"
                    if offload > 0.1
                    else f"KV: {kv_k}"
                )
                ax.annotate(
                    text,
                    (x, y),
                    xytext=(0, -10),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=COLORS["baseline"],
                    bbox=bbox_style,
                )

            elif mode == "rans_fused":
                text = f"KV: {kv_k}"
                ax.annotate(
                    text,
                    (x, y),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=COLORS["rans_fused"],
                    bbox=bbox_style,
                )

    g.map_dataframe(annotate_points)

    model_name = meta.get("baseline_model", "Model").split("/")[-1]
    g.fig.suptitle(
        f"Memory Wall: Throughput Scaling ({model_name})",
        y=1.05,
        fontweight="bold",
    )
    g.set_axis_labels(
        "VRAM Allocation Limit", "Throughput (Tokens / Sec)", fontweight="bold"
    )
    g.set_titles(
        col_template="Batch Size: {col_name}", row_template="Prompt: {row_name}"
    )

    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    if g._legend:
        g._legend.remove()

    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    new_labels = [LABELS.get(l, l) for l in labels]

    g.fig.legend(
        handles=handles,
        labels=new_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05), # Moved legend to bottom to save horizontal space
        ncol=3,
        title=None,
        frameon=True,
    )

    fname = f"throughput_wall_consolidated_{timestamp}.pdf"
    plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_offload_profile(df, meta, output_dir, timestamp):
    """
    NEW: PLOTS: CPU Offload Amount (GB) vs VRAM Utilization (X).
    Highlights the difference in PCIe traffic between baseline and rANS.
    """
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Extract unique offload amounts per config
    offload_df = df[["mode", "vram_util_config", "cpu_offload_gb"]].drop_duplicates()

    ax = sns.barplot(
        data=offload_df,
        x="vram_util_config",
        y="cpu_offload_gb",
        hue="mode",
        palette=COLORS,
        edgecolor="black",
        hue_order=["baseline", "rans_unfused", "rans_fused"],
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f GB", padding=3, rotation=90, fontsize=8)

    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("VRAM Allocation Limit", fontweight="bold")
    ax.set_ylabel("CPU Offload Size (GB)", fontweight="bold")

    model_name = meta.get("baseline_model", "Model").split("/")[-1]
    ax.set_title(f"CPU Parameter Offloading Profile: {model_name}", pad=15, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [LABELS.get(l, l) for l in labels]
    ax.legend(handles=handles, labels=new_labels, title=None, loc="upper right")

    fname = f"cpu_offload_profile_{timestamp}.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()
    print(f"Saved {fname}")


def plot_kv_density(df, meta, output_dir, timestamp):
    """PLOTS: KV Cache Tokens (Y) vs VRAM Utilization (X)."""
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    kv_df = df[["mode", "vram_util_config", "kv_cache_tokens"]].drop_duplicates()

    ax = sns.barplot(
        data=kv_df,
        x="vram_util_config",
        y="kv_cache_tokens",
        hue="mode",
        palette=COLORS,
        edgecolor="black",
        hue_order=["baseline", "rans_unfused", "rans_fused"],
    )

    for container in ax.containers:
        # Convert to 'k' format for readability if numbers are large
        ax.bar_label(container, labels=[f"{int(v.get_height())//1000}k" if v.get_height() > 0 else "0" for v in container], padding=3, rotation=90, fontsize=8)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("VRAM Allocation Limit", fontweight="bold")
    ax.set_ylabel("Available KV Cache (Tokens)", fontweight="bold")

    model_name = meta.get("baseline_model", "Model").split("/")[-1]
    ax.set_title(f"Memory Density Analysis: {model_name}", pad=15, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [LABELS.get(l, l) for l in labels]
    ax.legend(handles=handles, labels=new_labels, title=
