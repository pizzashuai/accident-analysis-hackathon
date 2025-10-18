#!/usr/bin/env python3
"""
Visualize and compare bounding box smoothing results.
Generates plots showing speed consistency and bbox stability metrics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_debug_data(debug_jsonl: Path) -> List[Dict]:
    """
    Load debug data from JSONL file.

    Args:
        debug_jsonl: Path to debug JSONL file

    Returns:
        List of debug records
    """
    data = []
    with open(debug_jsonl, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def organize_by_track(debug_data: List[Dict]) -> Dict[int, List[Dict]]:
    """
    Organize debug data by track ID.

    Args:
        debug_data: List of debug records

    Returns:
        Dictionary mapping track_id to list of records
    """
    tracks = {}
    for record in debug_data:
        track_id = record["track_id"]
        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append(record)

    # Sort each track by frame number
    for track_id in tracks:
        tracks[track_id].sort(key=lambda x: x["frame"])

    return tracks


def plot_speed_comparison(
    methods_data: Dict[str, List[Dict]], output_path: Path, track_id: int = None
):
    """
    Plot speed comparison across different smoothing methods.

    Args:
        methods_data: Dictionary mapping method name to debug data
        output_path: Path to save the plot
        track_id: Specific track ID to plot (None for first track)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_data)))

    for idx, (method, debug_data) in enumerate(methods_data.items()):
        tracks = organize_by_track(debug_data)

        if not tracks:
            continue

        # Use specified track_id or first available track
        if track_id is None or track_id not in tracks:
            track_id = list(tracks.keys())[0]

        records = tracks[track_id]

        frames = [r["frame"] for r in records]
        speeds = [r["speed_calc"]["smoothed_speed_mph"] for r in records]
        bbox_widths = [r["bbox_size"]["width"] for r in records]

        # Plot speeds
        axes[0].plot(frames, speeds, label=method, color=colors[idx], linewidth=2)

        # Plot bbox widths
        axes[1].plot(frames, bbox_widths, label=method, color=colors[idx], linewidth=2)

    axes[0].set_xlabel("Frame", fontsize=12)
    axes[0].set_ylabel("Speed (mph)", fontsize=12)
    axes[0].set_title(
        f"Speed Comparison - Track {track_id}", fontsize=14, fontweight="bold"
    )
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Frame", fontsize=12)
    axes[1].set_ylabel("Bounding Box Width (px)", fontsize=12)
    axes[1].set_title(
        f"Bounding Box Width - Track {track_id}", fontsize=14, fontweight="bold"
    )
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved speed comparison plot to: {output_path}")
    plt.close()


def plot_speed_variability(
    methods_data: Dict[str, List[Dict]], output_path: Path, track_id: int = None
):
    """
    Plot frame-to-frame speed changes to visualize smoothness.

    Args:
        methods_data: Dictionary mapping method name to debug data
        output_path: Path to save the plot
        track_id: Specific track ID to plot (None for first track)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_data)))

    for idx, (method, debug_data) in enumerate(methods_data.items()):
        tracks = organize_by_track(debug_data)

        if not tracks:
            continue

        # Use specified track_id or first available track
        if track_id is None or track_id not in tracks:
            track_id = list(tracks.keys())[0]

        records = tracks[track_id]

        if len(records) < 2:
            continue

        frames = []
        speed_changes = []

        for i in range(1, len(records)):
            prev_speed = records[i - 1]["speed_calc"]["smoothed_speed_mph"]
            curr_speed = records[i]["speed_calc"]["smoothed_speed_mph"]
            speed_change = abs(curr_speed - prev_speed)

            frames.append(records[i]["frame"])
            speed_changes.append(speed_change)

        ax.plot(frames, speed_changes, label=method, color=colors[idx], linewidth=2)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Speed Change (mph)", fontsize=12)
    ax.set_title(
        f"Frame-to-Frame Speed Variability - Track {track_id}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved speed variability plot to: {output_path}")
    plt.close()


def plot_bbox_stability(
    methods_data: Dict[str, List[Dict]], output_path: Path, track_id: int = None
):
    """
    Plot bounding box size stability.

    Args:
        methods_data: Dictionary mapping method name to debug data
        output_path: Path to save the plot
        track_id: Specific track ID to plot (None for first track)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_data)))

    for idx, (method, debug_data) in enumerate(methods_data.items()):
        tracks = organize_by_track(debug_data)

        if not tracks:
            continue

        # Use specified track_id or first available track
        if track_id is None or track_id not in tracks:
            track_id = list(tracks.keys())[0]

        records = tracks[track_id]

        if len(records) < 2:
            continue

        frames = []
        width_changes = []
        height_changes = []

        for i in range(1, len(records)):
            prev_width = records[i - 1]["bbox_size"]["width"]
            prev_height = records[i - 1]["bbox_size"]["height"]
            curr_width = records[i]["bbox_size"]["width"]
            curr_height = records[i]["bbox_size"]["height"]

            frames.append(records[i]["frame"])
            width_changes.append(abs(curr_width - prev_width))
            height_changes.append(abs(curr_height - prev_height))

        axes[0].plot(
            frames, width_changes, label=method, color=colors[idx], linewidth=2
        )
        axes[1].plot(
            frames, height_changes, label=method, color=colors[idx], linewidth=2
        )

    axes[0].set_xlabel("Frame", fontsize=12)
    axes[0].set_ylabel("Width Change (px)", fontsize=12)
    axes[0].set_title(
        f"Bounding Box Width Stability - Track {track_id}",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Frame", fontsize=12)
    axes[1].set_ylabel("Height Change (px)", fontsize=12)
    axes[1].set_title(
        f"Bounding Box Height Stability - Track {track_id}",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved bbox stability plot to: {output_path}")
    plt.close()


def plot_summary_metrics(results_json: Path, output_path: Path):
    """
    Plot summary bar charts comparing all methods.

    Args:
        results_json: Path to results JSON file
        output_path: Path to save the plot
    """
    with open(results_json, "r") as f:
        results = json.load(f)

    # Extract data
    methods = []
    avg_speeds = []
    std_speeds = []
    avg_speed_changes = []
    avg_bbox_changes = []

    for method, data in results.items():
        if data.get("success", False):
            methods.append(method)
            stats = data["stats"]
            avg_speeds.append(stats.get("avg_speed_mph", 0))
            std_speeds.append(stats.get("std_speed_mph", 0))
            avg_speed_changes.append(stats.get("avg_speed_change_mph", 0))
            avg_bbox_changes.append(stats.get("avg_bbox_size_change_px", 0))

    if not methods:
        print("No successful results to plot")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Average Speed
    axes[0, 0].bar(methods, avg_speeds, color="skyblue")
    axes[0, 0].set_ylabel("Speed (mph)", fontsize=11)
    axes[0, 0].set_title("Average Speed", fontsize=12, fontweight="bold")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Plot 2: Speed Standard Deviation (lower is better)
    axes[0, 1].bar(methods, std_speeds, color="lightcoral")
    axes[0, 1].set_ylabel("Std Dev (mph)", fontsize=11)
    axes[0, 1].set_title(
        "Speed Standard Deviation (lower = better)", fontsize=12, fontweight="bold"
    )
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Plot 3: Average Speed Change (lower is better)
    axes[1, 0].bar(methods, avg_speed_changes, color="lightgreen")
    axes[1, 0].set_ylabel("Speed Change (mph)", fontsize=11)
    axes[1, 0].set_title(
        "Avg Frame-to-Frame Speed Change (lower = smoother)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Plot 4: Average BBox Size Change (lower is better)
    axes[1, 1].bar(methods, avg_bbox_changes, color="plum")
    axes[1, 1].set_ylabel("Size Change (px)", fontsize=11)
    axes[1, 1].set_title(
        "Avg BBox Size Change (lower = more stable)", fontsize=12, fontweight="bold"
    )
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved summary metrics plot to: {output_path}")
    plt.close()


def main():
    """Main function to visualize bbox smoothing comparison."""
    comparison_dir = Path("out/bbox_smoothing_comparison")

    if not comparison_dir.exists():
        print(f"Error: Comparison directory not found: {comparison_dir}")
        print("Please run test_bbox_smoothing.py first.")
        sys.exit(1)

    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        comparison_dir = Path(sys.argv[1])

    # Load results
    results_json = comparison_dir / "smoothing_comparison_results.json"
    if not results_json.exists():
        print(f"Error: Results file not found: {results_json}")
        sys.exit(1)

    print("=" * 80)
    print("VISUALIZING BOUNDING BOX SMOOTHING COMPARISON")
    print("=" * 80)
    print(f"Comparison directory: {comparison_dir}")
    print()

    # Load debug data for each method
    methods_data = {}
    with open(results_json, "r") as f:
        results = json.load(f)

    for method, data in results.items():
        if data.get("success", False):
            debug_jsonl = Path(data["debug_jsonl"])
            if debug_jsonl.exists():
                methods_data[method] = load_debug_data(debug_jsonl)
                print(
                    f"Loaded {len(methods_data[method])} records for method: {method}"
                )

    if not methods_data:
        print("No debug data found to visualize")
        sys.exit(1)

    print()

    # Generate plots
    print("Generating plots...")
    print()

    plot_speed_comparison(methods_data, comparison_dir / "plot_speed_comparison.png")

    plot_speed_variability(methods_data, comparison_dir / "plot_speed_variability.png")

    plot_bbox_stability(methods_data, comparison_dir / "plot_bbox_stability.png")

    plot_summary_metrics(results_json, comparison_dir / "plot_summary_metrics.png")

    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {comparison_dir}")
    print()


if __name__ == "__main__":
    main()
