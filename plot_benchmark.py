# import json
# import glob
# import os
# import argparse
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# # --- PUBLICATION STYLE CONFIGURATION ---
# sns.set_theme(style="whitegrid")
# plt.rcParams.update(
#     {
#         "font.family": "serif",
#         "font.size": 10,  # Scaled down slightly for IEEE compliance
#         "axes.titlesize": 12,
#         "axes.labelsize": 10,
#         "xtick.labelsize": 8,
#         "ytick.labelsize": 8,
#         "legend.fontsize": 8,
#         "lines.linewidth": 2.0,
#         "lines.markersize": 6,
#         "pdf.fonttype": 42,
#         "ps.fonttype": 42,
#     }
# )

COLORS = {
    "baseline": "#82c8f0",  # Blue
    "rans_uncoal": "#FFDCA5",  # Orange
    "rans_coal": "#7DCDBE",  # Green
    "triton_baseline": "#B4A7D6",  # Purple
    "error": "#F5A5C8",  # Red
}

# LABELS = {
#     "baseline": "Baseline (cuBLAS)",
#     "rans_unfused": "rANS (Unfused)",
#     "rans_fused": "rANS (Fused Kernel)",
#     "triton_baseline": "Baseline (Triton)",
# }

LABELS = {
    "baseline": "FP16 Baseline",
    "rans_coal": "rANS (Coalesced)",
    "rans_uncoal": "rANS (Uncoalesced)",
    "triton_baseline": "Triton Base",
}

# # Standard IEEE Double Column Full Width is ~7.16 inches.
# # Half-page height is roughly 3.5 to 4.5 inches.
# FIG_WIDTH = 7.16
# FIG_HEIGHT = 3.5


# def load_data(filepath):
#     print(f"Loading {filepath}...")
#     with open(filepath, "r") as f:
#         raw = json.load(f)

#     if "results" in raw:
#         data_list = raw["results"]
#     elif "sweep_results" in raw and isinstance(raw["sweep_results"], list):
#         data_list = raw["sweep_results"]
#     else:
#         print("Warning: Detected legacy JSON format. Some sweep plots may fail.")
#         return None, raw["metadata"]

#     if not data_list:
#         raise ValueError("JSON file contains no result records.")

#     df = pd.DataFrame(data_list)
#     return df, raw["metadata"]


# # def plot_memory_wall(df, meta, output_dir, timestamp, target_batch_sizes=None):
# #     """
# #     PLOTS: Throughput (Y) vs VRAM Utilization (X).
# #     Consolidates subplots and annotates Offload/KV metrics directly on the points.
# #     """
# #     df_clean = df[df["avg_toks_sec"] > 0].copy()

# #     if target_batch_sizes:
# #         df_clean = df_clean[df_clean["batch_size"].isin(target_batch_sizes)]

# #     if df_clean.empty:
# #         print(f"No valid throughput data found for batch sizes: {target_batch_sizes}")
# #         return

# #     # Calculate optimal height to fit the 7.16 width requirement
# #     num_cols = len(df_clean["batch_size"].unique())
# #     aspect_ratio = (FIG_WIDTH / num_cols) / FIG_HEIGHT

# #     g = sns.relplot(
# #         data=df_clean,
# #         x="vram_util_config",
# #         y="avg_toks_sec",
# #         hue="mode",
# #         style="mode",
# #         col="batch_size",
# #         row="prompt_len",
# #         kind="line",
# #         palette=COLORS,
# #         markers=True,
# #         dashes=False,
# #         hue_order=["baseline", "rans_unfused", "rans_fused"],
# #         height=FIG_HEIGHT,
# #         aspect=aspect_ratio,
# #         facet_kws={"sharey": False},
# #     )

# #     def annotate_points(data, **kws):
# #         ax = plt.gca()
# #         agg_data = (
# #             data.groupby(["vram_util_config", "mode"])
# #             .agg(
# #                 {
# #                     "avg_toks_sec": "mean",
# #                     "cpu_offload_gb": "max",
# #                     "kv_cache_tokens": "max",
# #                 }
# #             )
# #             .reset_index()
# #         )

# #         for _, row in agg_data.iterrows():
# #             x = row["vram_util_config"]
# #             y = row["avg_toks_sec"]
# #             mode = row["mode"]
# #             offload = row["cpu_offload_gb"]
# #             kv = row["kv_cache_tokens"]

# #             kv_k = f"{int(kv)//1000}k" if kv > 0 else "0k"
# #             bbox_style = dict(
# #                 boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75
# #             )

# #             if mode == "baseline":
# #                 text = (
# #                     f"Off: {offload:.1f}G\nKV: {kv_k}"
# #                     if offload > 0.1
# #                     else f"KV: {kv_k}"
# #                 )
# #                 ax.annotate(
# #                     text,
# #                     (x, y),
# #                     xytext=(0, -10),
# #                     textcoords="offset points",
# #                     ha="center",
# #                     va="top",
# #                     fontsize=7,
# #                     color=COLORS["baseline"],
# #                     bbox=bbox_style,
# #                 )

# #             elif mode == "rans_fused":
# #                 text = f"KV: {kv_k}"
# #                 ax.annotate(
# #                     text,
# #                     (x, y),
# #                     xytext=(0, 10),
# #                     textcoords="offset points",
# #                     ha="center",
# #                     va="bottom",
# #                     fontsize=7,
# #                     color=COLORS["rans_fused"],
# #                     bbox=bbox_style,
# #                 )

# #     g.map_dataframe(annotate_points)

# #     model_name = meta.get("baseline_model", "Model").split("/")[-1]
# #     g.fig.suptitle(
# #         f"Memory Wall: Throughput Scaling ({model_name})",
# #         y=1.05,
# #         fontweight="bold",
# #     )
# #     g.set_axis_labels(
# #         "VRAM Allocation Limit", "Throughput (Tokens / Sec)", fontweight="bold"
# #     )
# #     g.set_titles(
# #         col_template="Batch Size: {col_name}", row_template="Prompt: {row_name}"
# #     )

# #     for ax in g.axes.flat:
# #         ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

# #     if g._legend:
# #         g._legend.remove()

# #     handles, labels = g.axes.flat[0].get_legend_handles_labels()
# #     new_labels = [LABELS.get(l, l) for l in labels]

# #     g.fig.legend(
# #         handles=handles,
# #         labels=new_labels,
# #         loc="upper center",
# #         bbox_to_anchor=(0.5, -0.05),  # Moved legend to bottom to save horizontal space
# #         ncol=3,
# #         title=None,
# #         frameon=True,
# #     )

# #     fname = f"throughput_wall_consolidated_{timestamp}.pdf"
# #     plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
# #     plt.close()
# #     print(f"Saved {fname}")


# # def plot_offload_profile(df, meta, output_dir, timestamp):
# #     """
# #     NEW: PLOTS: CPU Offload Amount (GB) vs VRAM Utilization (X).
# #     Highlights the difference in PCIe traffic between baseline and rANS.
# #     """
# #     plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

# #     # Extract unique offload amounts per config
# #     offload_df = df[["mode", "vram_util_config", "cpu_offload_gb"]].drop_duplicates()

# #     ax = sns.barplot(
# #         data=offload_df,
# #         x="vram_util_config",
# #         y="cpu_offload_gb",
# #         hue="mode",
# #         palette=COLORS,
# #         edgecolor="black",
# #         hue_order=["baseline", "rans_unfused", "rans_fused"],
# #     )

# #     for container in ax.containers:
# #         ax.bar_label(container, fmt="%.1f GB", padding=3, rotation=90, fontsize=8)

# #     ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
# #     ax.set_xlabel("VRAM Allocation Limit", fontweight="bold")
# #     ax.set_ylabel("CPU Offload Size (GB)", fontweight="bold")

# #     model_name = meta.get("baseline_model", "Model").split("/")[-1]
# #     ax.set_title(
# #         f"CPU Parameter Offloading Profile: {model_name}", pad=15, fontweight="bold"
# #     )

# #     handles, labels = ax.get_legend_handles_labels()
# #     new_labels = [LABELS.get(l, l) for l in labels]
# #     ax.legend(handles=handles, labels=new_labels, title=None, loc="upper right")

# #     fname = f"cpu_offload_profile_{timestamp}.pdf"
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(output_dir, fname))
# #     plt.close()
# #     print(f"Saved {fname}")


# # def plot_kv_density(df, meta, output_dir, timestamp):
# #     """PLOTS: KV Cache Tokens (Y) vs VRAM Utilization (X)."""
# #     plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
# #     kv_df = df[["mode", "vram_util_config", "kv_cache_tokens"]].drop_duplicates()

# #     ax = sns.barplot(
# #         data=kv_df,
# #         x="vram_util_config",
# #         y="kv_cache_tokens",
# #         hue="mode",
# #         palette=COLORS,
# #         edgecolor="black",
# #         hue_order=["baseline", "rans_unfused", "rans_fused"],
# #     )

# #     for container in ax.containers:
# #         # Convert to 'k' format for readability if numbers are large
# #         ax.bar_label(
# #             container,
# #             labels=[
# #                 f"{int(v.get_height())//1000}k" if v.get_height() > 0 else "0"
# #                 for v in container
# #             ],
# #             padding=3,
# #             rotation=90,
# #             fontsize=8,
# #         )

# #     ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
# #     ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
# #     ax.set_xlabel("VRAM Allocation Limit", fontweight="bold")
# #     ax.set_ylabel("Available KV Cache (Tokens)", fontweight="bold")

# #     model_name = meta.get("baseline_model", "Model").split("/")[-1]
# #     ax.set_title(f"Memory Density Analysis: {model_name}", pad=15, fontweight="bold")

# #     handles, labels = ax.get_legend_handles_labels()
# #     new_labels = [LABELS.get(l, l) for l in labels]
# #     ax.legend(handles=handles, labels=new_labels, title=None, loc="upper left")

# #     fname = f"kv_density_{timestamp}.pdf"
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(output_dir, fname))
# #     plt.close()
# #     print(f"Saved {fname}")


# # def plot_energy_landscape(df, meta, output_dir, timestamp):
# #     """PLOTS: Joules/Token (Y) vs VRAM Utilization (X)."""
# #     if "joules_per_token" not in df.columns:
# #         df["joules_per_token"] = df.apply(
# #             lambda row: row["energy_j"] / (row["batch_size"] * row["gen_len"])
# #             if row["energy_j"] > 0
# #             else 0,
# #             axis=1,
# #         )

# #     df_clean = df[df["energy_j"] > 0].copy()
# #     if df_clean.empty:
# #         return

# #     max_bs = df_clean["batch_size"].max()
# #     subset = df_clean[df_clean["batch_size"] == max_bs]

# #     plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
# #     ax = sns.barplot(
# #         data=subset,
# #         x="vram_util_config",
# #         y="joules_per_token",
# #         hue="mode",
# #         palette=COLORS,
# #         edgecolor="black",
# #         hue_order=["baseline", "rans_unfused", "rans_fused"],
# #     )

# #     ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
# #     ax.set_ylabel("Energy (Joules / Output Token) ↓", fontweight="bold")
# #     ax.set_xlabel("VRAM Allocation Limit", fontweight="bold")
# #     ax.set_title(f"Energy Efficiency @ Batch {max_bs}", pad=15, fontweight="bold")

# #     handles, labels = ax.get_legend_handles_labels()
# #     new_labels = [LABELS.get(l, l) for l in labels]
# #     ax.legend(handles=handles, labels=new_labels, loc="upper right")

# #     fname = f"energy_efficiency_{timestamp}.pdf"
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(output_dir, fname))
# #     plt.close()
# #     print(f"Saved {fname}")


# # def main():
# #     parser = argparse.ArgumentParser(description="Generate Publication Plots for rANS")
# #     parser.add_argument("--file", type=str, default="latest", help="JSON result file")
# #     parser.add_argument("--output", type=str, default=".", help="Output directory")
# #     parser.add_argument(
# #         "--plot_batch_sizes",
# #         type=str,
# #         default=None,
# #         help="Comma-separated batch sizes to plot (e.g., '1,4')",
# #     )
# #     args = parser.parse_args()

# #     if args.file == "latest":
# #         json_files = glob.glob("rans_sweep_*.json")
# #         if not json_files:
# #             print("No JSON files found.")
# #             return
# #         args.file = max(json_files, key=os.path.getmtime)

# #     df, meta = load_data(args.file)
# #     if df is None:
# #         return

# #     target_batch_sizes = None
# #     if args.plot_batch_sizes:
# #         target_batch_sizes = [int(x) for x in args.plot_batch_sizes.split(",")]

# #     ts_raw = meta.get("timestamp", "unknown")
# #     timestamp = ts_raw.replace(":", "").replace("-", "").split(".")[0]

# #     os.makedirs(args.output, exist_ok=True)
# #     print(f"Generating plots for: {args.file}")

# #     plot_memory_wall(df, meta, args.output, timestamp, target_batch_sizes)
# #     plot_offload_profile(df, meta, args.output, timestamp)
# #     plot_kv_density(df, meta, args.output, timestamp)
# #     plot_energy_landscape(df, meta, args.output, timestamp)

# #     print("\nDone. All plots generated.")


# # if __name__ == "__main__":
# #     main()

# import json
# import glob
# import os
# import argparse
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# # # --- PUBLICATION STYLE CONFIGURATION ---
# # sns.set_theme(style="whitegrid")
# # plt.rcParams.update(
# #     {
# #         "font.family": "serif",
# #         "font.size": 10,  # Scaled down slightly for IEEE compliance
# #         "axes.titlesize": 12,
# #         "axes.labelsize": 10,
# #         "xtick.labelsize": 8,
# #         "ytick.labelsize": 8,
# #         "legend.fontsize": 8,
# #         "lines.linewidth": 2.0,
# #         "lines.markersize": 6,
# #         "pdf.fonttype": 42,
# #         "ps.fonttype": 42,
# #     }
# # )

# # COLORS = {
# #     "baseline": "#82c8f0",  # Blue
# #     "triton_baseline": "#B4A7D6",  # Purple
# #     "rans_unfused": "#FFDCA5",  # Orange
# #     "rans_fused": "#7DCDBE",  # Green
# #     "error": "#F5A5C8",  # Red
# # }

# # LABELS = {
# #     "baseline": "Baseline (BF16)",
# #     "triton_baseline": "Baseline (Triton)",
# #     "rans_unfused": "rANS (Unfused)",
# #     "rans_fused": "rANS (Fused Kernel)",
# # }

# # # Standard IEEE Double Column Full Width is ~7.16 inches.
# # # Half-page height is roughly 3.5 to 4.5 inches.
# # FIG_WIDTH = 7.16
# # FIG_HEIGHT = 3.5


# # def load_data(filepath):
# #     print(f"Loading {filepath}...")
# #     with open(filepath, "r") as f:
# #         raw = json.load(f)

# #     if "results" in raw:
# #         data_list = raw["results"]
# #     elif "sweep_results" in raw and isinstance(raw["sweep_results"], list):
# #         data_list = raw["sweep_results"]
# #     else:
# #         print("Warning: Detected legacy JSON format. Some sweep plots may fail.")
# #         return None, raw["metadata"]

# #     if not data_list:
# #         raise ValueError("JSON file contains no result records.")

# #     df = pd.DataFrame(data_list)
# #     return df, raw["metadata"]


# # def plot_memory_wall(df, meta, output_dir, timestamp, target_batch_sizes=None):
# #     """
# #     PLOTS: Throughput (Y) vs VRAM Utilization (X).
# #     Consolidates subplots and annotates Offload/KV metrics directly on the points.
# #     """
# #     df_clean = df[df["avg_toks_sec"] > 0].copy()

# #     if target_batch_sizes:
# #         df_clean = df_clean[df_clean["batch_size"].isin(target_batch_sizes)]

# #     if df_clean.empty:
# #         print(f"No valid throughput data found for batch sizes: {target_batch_sizes}")
# #         return

# #     # Calculate optimal height to fit the 7.16 width requirement
# #     num_cols = len(df_clean["batch_size"].unique())
# #     aspect_ratio = (FIG_WIDTH / num_cols) / FIG_HEIGHT

# #     g = sns.relplot(
# #         data=df_clean,
# #         x="vram_util_config",
# #         y="avg_toks_sec",
# #         hue="mode",
# #         style="mode",
# #         col="batch_size",
# #         row="prompt_len",
# #         kind="line",
# #         palette=COLORS,
# #         markers=True,
# #         dashes=False,
# #         hue_order=["baseline", "triton_baseline", "rans_unfused", "rans_fused"],
# #         height=FIG_HEIGHT,
# #         aspect=aspect_ratio,
# #         facet_kws={"sharey": False},
# #     )

# #     def annotate_points(data, **kws):
# #         ax = plt.gca()
# #         agg_data = (
# #             data.groupby(["vram_util_config", "mode"])
# #             .agg(
# #                 {
# #                     "avg_toks_sec": "mean",
# #                     "cpu_offload_gb": "max",
# #                     "kv_cache_tokens": "max",
# #                 }
# #             )
# #             .reset_index()
# #         )

# #         for _, row in agg_data.iterrows():
# #             x = row["vram_util_config"]
# #             y = row["avg_toks_sec"]
# #             mode = row["mode"]
# #             offload = row["cpu_offload_gb"]
# #             kv = row["kv_cache_tokens"]

# #             kv_k = f"{int(kv)//1000}k" if kv > 0 else "0k"
# #             bbox_style = dict(
# #                 boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75
# #             )

# #             if mode in ["baseline", "triton_baseline"]:
# #                 text = (
# #                     f"Off: {offload:.1f}G\nKV: {kv_k}"
# #                     if offload > 0.1
# #                     else f"KV: {kv_k}"
# #                 )
# #                 ax.annotate(
# #                     text,
# #                     (x, y),
# #                     xytext=(0, -10),
# #                     textcoords="offset points",
# #                     ha="center",
# #                     va="top",
# #                     fontsize=7,
# #                     color=COLORS[mode],
# #                     bbox=bbox_style,
# #                 )

# #             elif mode == "rans_fused":
# #                 text = f"KV: {kv_k}"
# #                 ax.annotate(
# #                     text,
# #                     (x, y),
# #                     xytext=(0, 10),
# #                     textcoords="offset points",
# #                     ha="center",
# #                     va="bottom",
# #                     fontsize=7,
# #                     color=COLORS["rans_fused"],
# #                     bbox=bbox_style,
# #                 )

# #     g.map_dataframe(annotate_points)

# #     model_name = meta.get("baseline_model", "Model").split("/")[-1]
# #     g.fig.suptitle(
# #         f"Memory Wall: Throughput Scaling ({model_name})",
# #         y=1.05,
# #         fontweight="bold",
# #     )
# #     g.set_axis_labels(
# #         "VRAM Allocation Limit", "Throughput (Tokens / Sec)", fontweight="bold"
# #     )
# #     g.set_titles(
# #         col_template="Batch Size: {col_name}", row_template="Prompt: {row_name}"
# #     )

# #     for ax in g.axes.flat:
# #         ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

# #     if g._legend:
# #         g._legend.remove()

# #     handles, labels = g.axes.flat[0].get_legend_handles_labels()
# #     new_labels = [LABELS.get(l, l) for l in labels]

# #     g.fig.legend(
# #         handles=handles,
# #         labels=new_labels,
# #         loc="upper center",
# #         bbox_to_anchor=(0.5, -0.05),
# #         ncol=4,
# #         title=None,
# #         frameon=True,
# #     )

# #     fname = f"throughput_wall_consolidated_{timestamp}.pdf"
# #     plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
# #     plt.close()
# #     print(f"Saved {fname}")


# # def plot_offload_profile(df, meta, output_dir, timestamp):
# #     """
# #     NEW: PLOTS: CPU Offload Amount (GB) vs VRAM Utilization (X).
# #     Highlights the difference in PCIe traffic between baseline and rANS.
# #     """
# #     plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

# #     # Extract unique offload amounts per config
# #     offload_df = df[["mode", "vram_util_config", "cpu_offload_gb"]].drop_duplicates()

# #     ax = sns.barplot(
# #         data=offload_df,
# #         x="vram_util_config",
# #         y="cpu_offload_gb",
# #         hue="mode",
# #         palette=COLORS,
# #         edgecolor="black",
# #         hue_order=["baseline", "triton_baseline", "rans_unfused", "rans_fused"],
# #     )

# #     for container in ax.containers:
# #         ax.bar_label(container, fmt="%.1f GB", padding=3, rotation=90, fontsize=8)

# #     ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
# #     ax.set_xlabel("VRAM Allocation Limit", fontweight="bold")
# #     ax.set_ylabel("CPU Offload Size (GB)", fontweight="bold")

# #     model_name = meta.get("baseline_model", "Model").split("/")[-1]
# #     ax.set_title(
# #         f"CPU Parameter Offloading Profile: {model_name}", pad=15, fontweight="bold"
# #     )

# #     handles, labels = ax.get_legend_handles_labels()
# #     new_labels = [LABELS.get(l, l) for l in labels]
# #     ax.legend(handles=handles, labels=new_labels, title=None, loc="upper right")

# #     fname = f"cpu_offload_profile_{timestamp}.pdf"
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(output_dir, fname))
# #     plt.close()
# #     print(f"Saved {fname}")


# # def plot_kv_density(df, meta, output_dir, timestamp):
# #     """PLOTS: KV Cache Tokens (Y) vs VRAM Utilization (X)."""
# #     plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
# #     kv_df = df[["mode", "vram_util_config", "kv_cache_tokens"]].drop_duplicates()

# #     ax = sns.barplot(
# #         data=kv_df,
# #         x="vram_util_config",
# #         y="kv_cache_tokens",
# #         hue="mode",
# #         palette=COLORS,
# #         edgecolor="black",
# #         hue_order=["baseline", "triton_baseline", "rans_unfused", "rans_fused"],
# #     )

# #     for container in ax.containers:
# #         ax.bar_label(
# #             container,
# #             labels=[
# #                 f"{int(v.get_height())//1000}k" if v.get_height() > 0 else "0"
# #                 for v in container
# #             ],
# #             padding=3,
# #             rotation=90,
# #             fontsize=8,
# #         )

# #     ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
# #     ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
# #     ax.set_xlabel("VRAM Allocation Limit", fontweight="bold")
# #     ax.set_ylabel("Available KV Cache (Tokens)", fontweight="bold")

# #     model_name = meta.get("baseline_model", "Model").split("/")[-1]
# #     ax.set_title(f"Memory Density Analysis: {model_name}", pad=15, fontweight="bold")

# #     handles, labels = ax.get_legend_handles_labels()
# #     new_labels = [LABELS.get(l, l) for l in labels]
# #     ax.legend(handles=handles, labels=new_labels, title=None, loc="upper left")

# #     fname = f"kv_density_{timestamp}.pdf"
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(output_dir, fname))
# #     plt.close()
# #     print(f"Saved {fname}")


# # def plot_energy_landscape(df, meta, output_dir, timestamp):
# #     """PLOTS: Joules/Token (Y) vs VRAM Utilization (X)."""

# #     # Calculate Total Energy by adapting to either the new split metrics or the old unified metric
# #     if "gpu_energy_j" in df.columns:
# #         df["total_energy_j"] = (
# #             df["gpu_energy_j"]
# #             + df.get("cpu_energy_j", 0.0)
# #             + df.get("ram_energy_j", 0.0)
# #         )
# #     elif "energy_j" in df.columns:
# #         df["total_energy_j"] = df["energy_j"]
# #     else:
# #         print("Warning: No energy metrics found in dataframe. Skipping energy plot.")
# #         return

# #     if "joules_per_token" not in df.columns:
# #         df["joules_per_token"] = df.apply(
# #             lambda row: row["total_energy_j"] / (row["batch_size"] * row["gen_len"])
# #             if row["total_energy_j"] > 0
# #             else 0,
# #             axis=1,
# #         )

# #     df_clean = df[df["total_energy_j"] > 0].copy()
# #     if df_clean.empty:
# #         return

# #     max_bs = df_clean["batch_size"].max()
# #     subset = df_clean[df_clean["batch_size"] == max_bs]

# #     plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
# #     ax = sns.barplot(
# #         data=subset,
# #         x="vram_util_config",
# #         y="joules_per_token",
# #         hue="mode",
# #         palette=COLORS,
# #         edgecolor="black",
# #         hue_order=["baseline", "triton_baseline", "rans_unfused", "rans_fused"],
# #     )

# #     ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
# #     ax.set_ylabel("Energy (Joules / Output Token) ↓", fontweight="bold")
# #     ax.set_xlabel("VRAM Allocation Limit", fontweight="bold")
# #     ax.set_title(f"Energy Efficiency @ Batch {max_bs}", pad=15, fontweight="bold")

# #     handles, labels = ax.get_legend_handles_labels()
# #     new_labels = [LABELS.get(l, l) for l in labels]
# #     ax.legend(handles=handles, labels=new_labels, loc="upper right")

# #     fname = f"energy_efficiency_{timestamp}.pdf"
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(output_dir, fname))
# #     plt.close()
# #     print(f"Saved {fname}")

# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.ticker as ticker
# import numpy as np

# # --- IEEE Publication Settings ---
# FIG_WIDTH = 3.5
# FIG_HEIGHT = 2.8
# FONT_SIZE = 8

# # Enforce IEEE-compliant styling globally
# plt.rcParams.update(
#     {
#         "font.family": "serif",
#         "font.size": FONT_SIZE,
#         "axes.labelsize": FONT_SIZE,
#         "axes.titlesize": FONT_SIZE,
#         "xtick.labelsize": FONT_SIZE - 1,
#         "ytick.labelsize": FONT_SIZE - 1,
#         "legend.fontsize": FONT_SIZE - 2,
#         "lines.linewidth": 1.5,
#         "lines.markersize": 5,
#         "axes.grid": True,
#         "grid.alpha": 0.3,
#         "grid.linestyle": "--",
#     }
# )

# # Define your standard configurations (ensure COLORS and LABELS dictionaries exist in your scope)
# ALL_MODES = ["baseline", "triton_baseline", "rans_unfused", "rans_fused"]


# def get_present_modes(df):
#     """Returns only the modes that actually have data to prevent empty legend/bar gaps."""
#     return [m for m in ALL_MODES if m in df["mode"].unique()]


# def convert_vram_to_gb(df, meta):
#     """Converts the config percentage (e.g. 0.9) to actual physical GB limits."""
#     # Attempt to fetch true VRAM from meta, fallback to 80.0 GB (A100/H100) if missing
#     total_vram = meta.get("total_vram_gb", 47.97)
#     df["vram_limit_gb"] = df["vram_util_config"] * total_vram
#     return df


# # def plot_memory_wall(df, meta, output_dir, timestamp, target_batch_sizes=None):
# #     """
# #     PLOTS: Throughput (Y) vs VRAM Limit in GB (X).
# #     Aggregates across prompt lengths and outputs one IEEE-ready plot per batch size.
# #     """
# #     df_clean = df[df["avg_toks_sec"] > 0].copy()
# #     if target_batch_sizes:
# #         df_clean = df_clean[df_clean["batch_size"].isin(target_batch_sizes)]

# #     if df_clean.empty:
# #         print(f"No valid throughput data found for batch sizes: {target_batch_sizes}")
# #         return

# #     df_clean = convert_vram_to_gb(df_clean, meta)
# #     present_modes = get_present_modes(df_clean)

# #     # AGGREGATION: Average across prompt lengths for a "mixed workload" metric
# #     agg_data = (
# #         df_clean.groupby(["batch_size", "vram_limit_gb", "mode"])
# #         .agg(
# #             {"avg_toks_sec": "mean", "cpu_offload_gb": "max", "kv_cache_tokens": "max"}
# #         )
# #         .reset_index()
# #     )

# #     # Generate one standalone 3.5-inch plot per batch size
# #     for bs in agg_data["batch_size"].unique():
# #         bs_df = agg_data[agg_data["batch_size"] == bs]

# #         fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# #         # Clear, thick lines
# #         sns.lineplot(
# #             data=bs_df,
# #             x="vram_limit_gb",
# #             y="avg_toks_sec",
# #             hue="mode",
# #             style="mode",
# #             markers=True,
# #             dashes=False,
# #             palette=COLORS,
# #             hue_order=present_modes,
# #             ax=ax,
# #         )

# #         # Annotate points
# #         for _, row in bs_df.iterrows():
# #             x = row["vram_limit_gb"]
# #             y = row["avg_toks_sec"]
# #             mode = row["mode"]
# #             offload = row["cpu_offload_gb"]
# #             kv = row["kv_cache_tokens"]

# #             kv_k = f"{int(kv)//1000}k" if kv > 0 else "0k"
# #             bbox_style = dict(
# #                 boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8
# #             )

# #             # Adjust font size to fit the tight 3.5 inch space
# #             if mode in ["baseline", "triton_baseline"]:
# #                 text = (
# #                     f"Off: {offload:.1f}G\nKV: {kv_k}"
# #                     if offload > 0.1
# #                     else f"KV: {kv_k}"
# #                 )
# #                 ax.annotate(
# #                     text,
# #                     (x, y),
# #                     xytext=(0, -8),
# #                     textcoords="offset points",
# #                     ha="center",
# #                     va="top",
# #                     fontsize=5.5,
# #                     color=COLORS[mode],
# #                     bbox=bbox_style,
# #                 )
# #             elif mode == "rans_fused":
# #                 text = f"KV: {kv_k}"
# #                 ax.annotate(
# #                     text,
# #                     (x, y),
# #                     xytext=(0, 8),
# #                     textcoords="offset points",
# #                     ha="center",
# #                     va="bottom",
# #                     fontsize=5.5,
# #                     color=COLORS[mode],
# #                     bbox=bbox_style,
# #                 )

# #         # Formatting
# #         ax.set_xlabel("VRAM Available (GB)", fontweight="bold")
# #         ax.set_ylabel("Avg Throughput (Tokens / Sec)", fontweight="bold")
# #         ax.xaxis.set_major_formatter(
# #             ticker.FormatStrFormatter("%g")
# #         )  # Clean integer/float formatting
# #         ax.set_title("")  # Removed redundant title

# #         # Legend above the plot
# #         handles, labels = ax.get_legend_handles_labels()
# #         new_labels = [LABELS.get(l, l) for l in labels]
# #         ax.legend(
# #             handles=handles,
# #             labels=new_labels,
# #             loc="upper center",
# #             bbox_to_anchor=(0.5, 1.25),
# #             ncol=2,
# #             frameon=False,
# #             columnspacing=1.0,
# #         )

# #         plt.subplots_adjust(top=0.8, bottom=0.15, left=0.15, right=0.95)
# #         fname = f"throughput_wall_bs{bs}_{timestamp}.pdf"
# #         plt.savefig(os.path.join(output_dir, fname), format="pdf", dpi=300)
# #         plt.close()
# #         print(f"Saved {fname}")

# from adjustText import adjust_text

# def plot_memory_wall(df, meta, output_dir, timestamp, target_batch_sizes=None):
#     """
#     PLOTS: Stacked panels for Throughput, KV Cache, and CPU Offload vs VRAM.
#     Zero text annotations. Uses a shared X-axis for perfect vertical correlation.
#     """
#     df_clean = df[df["avg_toks_sec"] > 0].copy()
#     if target_batch_sizes:
#         df_clean = df_clean[df_clean["batch_size"].isin(target_batch_sizes)]

#     if df_clean.empty:
#         print(f"No valid throughput data found for batch sizes: {target_batch_sizes}")
#         return

#     df_clean = convert_vram_to_gb(df_clean, meta)
#     present_modes = get_present_modes(df_clean)

#     # AGGREGATION: Average across prompt lengths for a "mixed workload" metric
#     agg_data = (
#         df_clean.groupby(["batch_size", "vram_limit_gb", "mode"])
#         .agg(
#             {"avg_toks_sec": "mean", "cpu_offload_gb": "max", "kv_cache_tokens": "max"}
#         )
#         .reset_index()
#     )

#     # Generate one standalone 3.5-inch wide plot per batch size
#     for bs in agg_data["batch_size"].unique():
#         bs_df = agg_data[agg_data["batch_size"] == bs]

#         # 3.5 inches wide, but taller (4.5 inches) to comfortably fit the 3 stacked panels
#         fig, axes = plt.subplots(
#             3, 1, figsize=(FIG_WIDTH, 4.5), sharex=True, gridspec_kw={'hspace': 0.15}
#         )

#         # --- PANEL 1: Throughput ---
#         sns.lineplot(
#             data=bs_df, x="vram_limit_gb", y="avg_toks_sec",
#             hue="mode", style="mode", markers=True, dashes=False,
#             palette=COLORS, hue_order=present_modes, ax=axes[0], legend=False
#         )
#         axes[0].set_ylabel("Throughput\n(Tok/s)", fontweight="bold")

#         # --- PANEL 2: KV Cache Tokens ---
#         sns.lineplot(
#             data=bs_df, x="vram_limit_gb", y="kv_cache_tokens",
#             hue="mode", style="mode", markers=True, dashes=False,
#             palette=COLORS, hue_order=present_modes, ax=axes[1], legend=False
#         )
#         axes[1].set_ylabel("KV Cache\n(Tokens)", fontweight="bold")
#         # Format Y-axis to show 'k' for thousands (e.g., 32k)
#         axes[1].yaxis.set_major_formatter(
#             ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
#         )

#         # --- PANEL 3: CPU Offload ---
#         sns.lineplot(
#             data=bs_df, x="vram_limit_gb", y="cpu_offload_gb",
#             hue="mode", style="mode", markers=True, dashes=False,
#             palette=COLORS, hue_order=present_modes, ax=axes[2]
#         )
#         axes[2].set_ylabel("Offload\n(GB)", fontweight="bold")
#         axes[2].set_xlabel("VRAM Available (GB)", fontweight="bold")
#         axes[2].xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))

#         # Strip the automatic legend from the bottom plot
#         if axes[2].get_legend():
#             handles, labels = axes[2].get_legend_handles_labels()
#             axes[2].get_legend().remove()
#             new_labels = [LABELS.get(l, l) for l in labels]

#             # Place ONE unified legend at the very top of the entire figure
#             fig.legend(
#                 handles, new_labels,
#                 loc="upper center",
#                 bbox_to_anchor=(0.5, 1.05),
#                 ncol=2,
#                 frameon=False,
#                 columnspacing=1.0
#             )

#         # Apply grid to all panels for readability
#         for ax in axes:
#             ax.grid(True, alpha=0.3, linestyle="--")

#         # bbox_inches="tight" ensures the Y-axis labels don't get clipped when saving
#         fname = f"throughput_wall_bs{bs}_{timestamp}.pdf"
#         plt.savefig(os.path.join(output_dir, fname), format="pdf", dpi=300, bbox_inches="tight")
#         plt.close()
#         print(f"Saved {fname}")

# def plot_offload_profile(df, meta, output_dir, timestamp):
#     """
#     PLOTS: CPU Offload Amount (GB) vs VRAM Allocation (GB).
#     """
#     df = convert_vram_to_gb(df, meta)
#     present_modes = get_present_modes(df)

#     # Aggregate: Max offload for a given config across all prompts/batches
#     offload_df = (
#         df.groupby(["vram_limit_gb", "mode"])["cpu_offload_gb"].max().reset_index()
#     )

#     fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

#     # linewidth=0.8 keeps bar borders crisp at this scale
#     sns.barplot(
#         data=offload_df,
#         x="vram_limit_gb",
#         y="cpu_offload_gb",
#         hue="mode",
#         palette=COLORS,
#         edgecolor="black",
#         hue_order=present_modes,
#         ax=ax,
#         linewidth=0.8,
#     )

#     for container in ax.containers:
#         ax.bar_label(container, fmt="%.1f G", padding=2, rotation=90, fontsize=6)

#     ax.set_xlabel("VRAM Available (GB)", fontweight="bold")
#     ax.set_ylabel("Max CPU Offload Size (GB)", fontweight="bold")
#     ax.set_title("")

#     handles, labels = ax.get_legend_handles_labels()
#     new_labels = [LABELS.get(l, l) for l in labels]
#     ax.legend(
#         handles=handles,
#         labels=new_labels,
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.25),
#         ncol=2,
#         frameon=False,
#         columnspacing=1.0,
#     )

#     plt.subplots_adjust(top=0.8, bottom=0.15, left=0.15, right=0.95)
#     fname = f"cpu_offload_profile_{timestamp}.pdf"
#     plt.savefig(os.path.join(output_dir, fname), format="pdf", dpi=300)
#     plt.close()
#     print(f"Saved {fname}")


# def plot_kv_density(df, meta, output_dir, timestamp):
#     """PLOTS: KV Cache Tokens vs VRAM Allocation (GB)."""
#     df = convert_vram_to_gb(df, meta)
#     present_modes = get_present_modes(df)

#     kv_df = df.groupby(["vram_limit_gb", "mode"])["kv_cache_tokens"].max().reset_index()

#     fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
#     sns.barplot(
#         data=kv_df,
#         x="vram_limit_gb",
#         y="kv_cache_tokens",
#         hue="mode",
#         palette=COLORS,
#         edgecolor="black",
#         hue_order=present_modes,
#         ax=ax,
#         linewidth=0.8,
#     )

#     for container in ax.containers:
#         ax.bar_label(
#             container,
#             labels=[
#                 f"{int(v.get_height())//1000}k" if v.get_height() > 0 else "0"
#                 for v in container
#             ],
#             padding=2,
#             rotation=90,
#             fontsize=6,
#         )

#     ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
#     ax.set_xlabel("VRAM Available (GB)", fontweight="bold")
#     ax.set_ylabel("Max Available KV Cache (Tokens)", fontweight="bold")
#     ax.set_title("")

#     handles, labels = ax.get_legend_handles_labels()
#     new_labels = [LABELS.get(l, l) for l in labels]
#     ax.legend(
#         handles=handles,
#         labels=new_labels,
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.25),
#         ncol=2,
#         frameon=False,
#         columnspacing=1.0,
#     )

#     plt.subplots_adjust(top=0.8, bottom=0.15, left=0.15, right=0.95)
#     fname = f"kv_density_{timestamp}.pdf"
#     plt.savefig(os.path.join(output_dir, fname), format="pdf", dpi=300)
#     plt.close()
#     print(f"Saved {fname}")


# def plot_energy_landscape(df, meta, output_dir, timestamp):
#     """PLOTS: Joules/Token vs VRAM Allocation (GB)."""

#     # Adapt to energy metrics
#     if "gpu_energy_j" in df.columns:
#         df["total_energy_j"] = (
#             df["gpu_energy_j"]
#             + df.get("cpu_energy_j", 0.0)
#             + df.get("ram_energy_j", 0.0)
#         )
#     elif "energy_j" in df.columns:
#         df["total_energy_j"] = df["energy_j"]
#     else:
#         print("Warning: No energy metrics found in dataframe. Skipping energy plot.")
#         return

#     if "joules_per_token" not in df.columns:
#         df["joules_per_token"] = df.apply(
#             lambda row: row["total_energy_j"] / (row["batch_size"] * row["gen_len"])
#             if row["total_energy_j"] > 0
#             else 0,
#             axis=1,
#         )

#     df_clean = df[df["total_energy_j"] > 0].copy()
#     if df_clean.empty:
#         return

#     df_clean = convert_vram_to_gb(df_clean, meta)
#     present_modes = get_present_modes(df_clean)

#     # Aggregate: Average energy per token across prompt lengths
#     # We focus on the maximum batch size to highlight heavy load efficiency
#     max_bs = df_clean["batch_size"].max()
#     subset = df_clean[df_clean["batch_size"] == max_bs]
#     energy_df = (
#         subset.groupby(["vram_limit_gb", "mode"])["joules_per_token"]
#         .mean()
#         .reset_index()
#     )

#     fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
#     sns.barplot(
#         data=energy_df,
#         x="vram_limit_gb",
#         y="joules_per_token",
#         hue="mode",
#         palette=COLORS,
#         edgecolor="black",
#         hue_order=present_modes,
#         ax=ax,
#         linewidth=0.8,
#     )

#     ax.set_ylabel("Energy (Joules / Token) ↓", fontweight="bold")
#     ax.set_xlabel("VRAM Available (GB)", fontweight="bold")
#     ax.set_title("")

#     handles, labels = ax.get_legend_handles_labels()
#     new_labels = [LABELS.get(l, l) for l in labels]
#     ax.legend(
#         handles=handles,
#         labels=new_labels,
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.25),
#         ncol=2,
#         frameon=False,
#         columnspacing=1.0,
#     )

#     plt.subplots_adjust(top=0.8, bottom=0.15, left=0.15, right=0.95)
#     fname = f"energy_efficiency_{timestamp}.pdf"
#     plt.savefig(os.path.join(output_dir, fname), format="pdf", dpi=300)
#     plt.close()
#     print(f"Saved {fname}")


# def main():
#     parser = argparse.ArgumentParser(description="Generate Publication Plots for rANS")
#     parser.add_argument("--file", type=str, default="latest", help="JSON result file")
#     parser.add_argument("--output", type=str, default=".", help="Output directory")
#     parser.add_argument(
#         "--plot_batch_sizes",
#         type=str,
#         default=None,
#         help="Comma-separated batch sizes to plot (e.g., '1,4')",
#     )
#     args = parser.parse_args()

#     if args.file == "latest":
#         json_files = glob.glob("rans_sweep_*.json")
#         if not json_files:
#             print("No JSON files found.")
#             return
#         args.file = max(json_files, key=os.path.getmtime)

#     df, meta = load_data(args.file)
#     if df is None:
#         return

#     target_batch_sizes = None
#     if args.plot_batch_sizes:
#         target_batch_sizes = [int(x) for x in args.plot_batch_sizes.split(",")]

#     ts_raw = meta.get("timestamp", "unknown")
#     timestamp = ts_raw.replace(":", "").replace("-", "").split(".")[0]

#     os.makedirs(args.output, exist_ok=True)
#     print(f"Generating plots for: {args.file}")

#     plot_memory_wall(df, meta, args.output, timestamp, target_batch_sizes)
#     plot_offload_profile(df, meta, args.output, timestamp)
#     plot_kv_density(df, meta, args.output, timestamp)
#     plot_energy_landscape(df, meta, args.output, timestamp)

#     print("\nDone. All plots generated.")


# if __name__ == "__main__":
#     main()


import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from adjustText import adjust_text

# --- IEEE STYLE GLOBALS ---
FIG_WIDTH = 3.5
FIG_HEIGHT = 2.5

# Distinguish the proposed Uncoalesced (Dark Blue) from the legacy Coalesced (Light Blue)
COLORS = {
    "baseline": "#2ca02c",  # Green
    "rans_coal": "#aec7e8",  # Light Blue (Ablation/Legacy)
    "rans_uncoal": "#1f77b4",  # Dark Blue (Proposed)
    "triton_baseline": "#ff7f0e",  # Orange
}

LABELS = {
    "baseline": "FP16 Baseline",
    "rans_coal": "rANS (Coalesced)",
    "rans_uncoal": "rANS (Uncoalesced)",
    "triton_baseline": "Triton Base",
}


# --- DATA HELPERS ---
def load_data(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data["results"])
        return df, data["metadata"]
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def convert_vram_to_gb(df, meta):
    if "vram_limit_gb" not in df.columns:
        df["vram_limit_gb"] = df["vram_util_config"] * df["total_vram_gb"]
    return df


def get_present_modes(df):
    # Enforce a strict visual drawing order: Baseline -> Coalesced -> Uncoalesced
    order = ["baseline", "triton_baseline", "rans_coal", "rans_uncoal"]
    return [m for m in order if m in df["mode"].unique()]


# --- PLOTTING FUNCTIONS ---
def plot_memory_wall(df, meta, output_dir, timestamp, target_batch_sizes=None):
    """
    PLOTS: Stacked panels for Throughput, KV Cache, and CPU Offload vs VRAM.
    """
    df_clean = df[df["avg_toks_sec"] > 0].copy()
    if target_batch_sizes:
        df_clean = df_clean[df_clean["batch_size"].isin(target_batch_sizes)]

    if df_clean.empty:
        print(f"No valid throughput data found for batch sizes: {target_batch_sizes}")
        return

    df_clean = convert_vram_to_gb(df_clean, meta)
    present_modes = get_present_modes(df_clean)

    agg_data = (
        df_clean.groupby(["batch_size", "vram_limit_gb", "mode"])
        .agg(
            {"avg_toks_sec": "mean", "cpu_offload_gb": "max", "kv_cache_tokens": "max"}
        )
        .reset_index()
    )

    for bs in agg_data["batch_size"].unique():
        bs_df = agg_data[agg_data["batch_size"] == bs]

        plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6})
        fig, axes = plt.subplots(
            3, 1, figsize=(FIG_WIDTH, 4.5), sharex=True, gridspec_kw={"hspace": 0.15}
        )

        # --- PANEL 1: Throughput ---
        sns.lineplot(
            data=bs_df,
            x="vram_limit_gb",
            y="avg_toks_sec",
            hue="mode",
            style="mode",
            markers=True,
            dashes=False,
            palette=COLORS,
            hue_order=present_modes,
            ax=axes[0],
            legend=False,
        )
        axes[0].set_ylabel("Throughput\n(Tok/s)", fontweight="bold")

        # --- PANEL 2: KV Cache Tokens ---
        sns.lineplot(
            data=bs_df,
            x="vram_limit_gb",
            y="kv_cache_tokens",
            hue="mode",
            style="mode",
            markers=True,
            dashes=False,
            palette=COLORS,
            hue_order=present_modes,
            ax=axes[1],
            legend=False,
        )
        axes[1].set_ylabel("KV Cache\n(Tokens)", fontweight="bold")
        axes[1].yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
            )
        )

        # --- PANEL 3: CPU Offload ---
        sns.lineplot(
            data=bs_df,
            x="vram_limit_gb",
            y="cpu_offload_gb",
            hue="mode",
            style="mode",
            markers=True,
            dashes=False,
            palette=COLORS,
            hue_order=present_modes,
            ax=axes[2],
        )
        axes[2].set_ylabel("Offload\n(GB)", fontweight="bold")
        axes[2].set_xlabel("VRAM Available (GB)", fontweight="bold")
        axes[2].xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))

        # Strip the automatic legend from the bottom plot
        if axes[2].get_legend():
            handles, labels = axes[2].get_legend_handles_labels()
            axes[2].get_legend().remove()
            new_labels = [LABELS.get(l, l) for l in labels]

            # Place ONE unified legend at the top. Updated to ncol=3 to fit Baseline, Coal, and Uncoal
            fig.legend(
                handles,
                new_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.05),
                ncol=3,
                frameon=False,
                columnspacing=0.8,
            )

        for ax in axes:
            ax.grid(True, alpha=0.3, linestyle="--")

        fname = f"throughput_wall_bs{bs}_{timestamp}.pdf"
        plt.savefig(
            os.path.join(output_dir, fname), format="pdf", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"Saved {fname}")


def plot_offload_profile(df, meta, output_dir, timestamp):
    """PLOTS: CPU Offload Amount (GB) vs VRAM Allocation (GB)."""
    df = convert_vram_to_gb(df, meta)
    present_modes = get_present_modes(df)

    offload_df = (
        df.groupby(["vram_limit_gb", "mode"])["cpu_offload_gb"].max().reset_index()
    )

    plt.rcParams.update({"font.size": 7, "legend.fontsize": 6})
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    sns.barplot(
        data=offload_df,
        x="vram_limit_gb",
        y="cpu_offload_gb",
        hue="mode",
        palette=COLORS,
        edgecolor="black",
        hue_order=present_modes,
        ax=ax,
        linewidth=0.8,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f G", padding=2, rotation=90, fontsize=5)

    ax.set_xlabel("VRAM Available (GB)", fontweight="bold")
    ax.set_ylabel("Max CPU Offload Size (GB)", fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [LABELS.get(l, l) for l in labels]
    ax.legend(
        handles=handles,
        labels=new_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
        columnspacing=0.8,
    )

    plt.subplots_adjust(top=0.8, bottom=0.15, left=0.15, right=0.95)
    fname = f"cpu_offload_profile_{timestamp}.pdf"
    plt.savefig(os.path.join(output_dir, fname), format="pdf", dpi=300)
    plt.close()
    print(f"Saved {fname}")


def plot_kv_density(df, meta, output_dir, timestamp):
    """PLOTS: KV Cache Tokens vs VRAM Allocation (GB)."""
    df = convert_vram_to_gb(df, meta)
    present_modes = get_present_modes(df)

    kv_df = df.groupby(["vram_limit_gb", "mode"])["kv_cache_tokens"].max().reset_index()

    plt.rcParams.update({"font.size": 7, "legend.fontsize": 6})
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    sns.barplot(
        data=kv_df,
        x="vram_limit_gb",
        y="kv_cache_tokens",
        hue="mode",
        palette=COLORS,
        edgecolor="black",
        hue_order=present_modes,
        ax=ax,
        linewidth=0.8,
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[
                f"{int(v.get_height())//1000}k" if v.get_height() > 0 else "0"
                for v in container
            ],
            padding=2,
            rotation=90,
            fontsize=5,
        )

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
    ax.set_xlabel("VRAM Available (GB)", fontweight="bold")
    ax.set_ylabel("Max Available KV Cache (Tokens)", fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [LABELS.get(l, l) for l in labels]
    ax.legend(
        handles=handles,
        labels=new_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
        columnspacing=0.8,
    )

    plt.subplots_adjust(top=0.8, bottom=0.15, left=0.15, right=0.95)
    fname = f"kv_density_{timestamp}.pdf"
    plt.savefig(os.path.join(output_dir, fname), format="pdf", dpi=300)
    plt.close()
    print(f"Saved {fname}")


def plot_energy_landscape(df, meta, output_dir, timestamp):
    """PLOTS: Joules/Token vs VRAM Allocation (GB)."""
    if "gpu_energy_j" in df.columns:
        df["total_energy_j"] = (
            df["gpu_energy_j"]
            + df.get("cpu_energy_j", 0.0)
            + df.get("ram_energy_j", 0.0)
        )
    elif "energy_j" in df.columns:
        df["total_energy_j"] = df["energy_j"]
    else:
        print("Warning: No energy metrics found in dataframe. Skipping energy plot.")
        return

    if "joules_per_token" not in df.columns:
        df["joules_per_token"] = df.apply(
            lambda row: row["total_energy_j"] / (row["batch_size"] * row["gen_len"])
            if row["total_energy_j"] > 0
            else 0,
            axis=1,
        )

    df_clean = df[df["total_energy_j"] > 0].copy()
    if df_clean.empty:
        return

    df_clean = convert_vram_to_gb(df_clean, meta)
    present_modes = get_present_modes(df_clean)

    max_bs = df_clean["batch_size"].max()
    subset = df_clean[df_clean["batch_size"] == max_bs]
    energy_df = (
        subset.groupby(["vram_limit_gb", "mode"])["joules_per_token"]
        .mean()
        .reset_index()
    )

    plt.rcParams.update({"font.size": 7, "legend.fontsize": 6})
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    sns.barplot(
        data=energy_df,
        x="vram_limit_gb",
        y="joules_per_token",
        hue="mode",
        palette=COLORS,
        edgecolor="black",
        hue_order=present_modes,
        ax=ax,
        linewidth=0.8,
    )

    ax.set_ylabel("Energy (Joules / Token) ↓", fontweight="bold")
    ax.set_xlabel("VRAM Available (GB)", fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [LABELS.get(l, l) for l in labels]
    ax.legend(
        handles=handles,
        labels=new_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
        columnspacing=0.8,
    )

    plt.subplots_adjust(top=0.8, bottom=0.15, left=0.15, right=0.95)
    fname = f"energy_efficiency_{timestamp}.pdf"
    plt.savefig(os.path.join(output_dir, fname), format="pdf", dpi=300)
    plt.close()
    print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="Generate Publication Plots for rANS")
    parser.add_argument("--file", type=str, default="latest", help="JSON result file")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    parser.add_argument(
        "--plot_batch_sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes to plot (e.g., '1,4')",
    )
    args = parser.parse_args()

    if args.file == "latest":
        json_files = glob.glob("rans_sweep_*.json")
        if not json_files:
            print("No JSON files found.")
            return
        args.file = max(json_files, key=os.path.getmtime)

    df, meta = load_data(args.file)
    if df is None:
        return

    target_batch_sizes = None
    if args.plot_batch_sizes:
        target_batch_sizes = [int(x) for x in args.plot_batch_sizes.split(",")]

    ts_raw = meta.get("timestamp", "unknown")
    timestamp = ts_raw.replace(":", "").replace("-", "").split(".")[0]

    os.makedirs(args.output, exist_ok=True)
    print(f"Generating plots for: {args.file}")

    plot_memory_wall(df, meta, args.output, timestamp, target_batch_sizes)
    plot_offload_profile(df, meta, args.output, timestamp)
    plot_kv_density(df, meta, args.output, timestamp)
    plot_energy_landscape(df, meta, args.output, timestamp)

    print("\nDone. All plots generated.")


if __name__ == "__main__":
    main()
