#!/usr/bin/env python3
"""
Visualize speed data from debug JSONL files.
Creates plots comparing raw vs smoothed speeds.
"""

import json
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def read_debug_jsonl(jsonl_path):
    """Read debug data from JSONL file."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def plot_track_speeds(track_data, track_id, output_path=None):
    """Plot raw vs smoothed speeds for a track."""
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for plotting")
        return

    frames = [d["frame"] for d in track_data]
    times = [d["time"] for d in track_data]
    raw_speeds = [d["speed_calc"]["raw_speed_mph"] for d in track_data]
    smoothed_speeds = [d["speed_calc"]["smoothed_speed_mph"] for d in track_data]
    smoothing_method = track_data[0]["speed_calc"]["smoothing_method"]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Speed vs Time
    ax1.plot(
        times, raw_speeds, "o-", alpha=0.5, label="Raw Speed", color="red", markersize=3
    )
    ax1.plot(
        times, smoothed_speeds, "o-", label="Smoothed Speed", color="blue", markersize=3
    )
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Speed (mph)")
    ax1.set_title(f"Track {track_id} - Speed Comparison ({smoothing_method})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Speed vs Frame
    ax2.plot(
        frames,
        raw_speeds,
        "o-",
        alpha=0.5,
        label="Raw Speed",
        color="red",
        markersize=3,
    )
    ax2.plot(
        frames,
        smoothed_speeds,
        "o-",
        label="Smoothed Speed",
        color="blue",
        markersize=3,
    )
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("Speed (mph)")
    ax2.set_title(f"Track {track_id} - Speed by Frame")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_bbox_size_vs_speed(track_data, track_id, output_path=None):
    """Plot bbox size variations vs speed variations."""
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for plotting")
        return

    frames = [d["frame"] for d in track_data]
    widths = [d["bbox_size"]["width"] for d in track_data]
    heights = [d["bbox_size"]["height"] for d in track_data]
    raw_speeds = [d["speed_calc"]["raw_speed_mph"] for d in track_data]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Bbox width over time
    ax1.plot(frames, widths, "o-", color="green", markersize=3)
    ax1.set_ylabel("Bbox Width (pixels)")
    ax1.set_title(f"Track {track_id} - Bounding Box Size Variations")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bbox height over time
    ax2.plot(frames, heights, "o-", color="orange", markersize=3)
    ax2.set_ylabel("Bbox Height (pixels)")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Raw speed over time
    ax3.plot(frames, raw_speeds, "o-", color="red", markersize=3)
    ax3.set_xlabel("Frame Number")
    ax3.set_ylabel("Raw Speed (mph)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison_multiple_methods(debug_files, track_id, output_path=None):
    """Compare the same track across different smoothing methods."""
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for plotting")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    colors = ["red", "blue", "green", "purple", "orange"]

    for idx, (method_name, file_path) in enumerate(debug_files.items()):
        data = read_debug_jsonl(file_path)

        # Filter for specific track
        track_data = [d for d in data if d["track_id"] == track_id]
        if not track_data:
            continue

        frames = [d["frame"] for d in track_data]
        raw_speeds = [d["speed_calc"]["raw_speed_mph"] for d in track_data]
        smoothed_speeds = [d["speed_calc"]["smoothed_speed_mph"] for d in track_data]

        color = colors[idx % len(colors)]

        # Plot raw speeds (only once, they should be the same)
        if idx == 0:
            ax1.plot(
                frames,
                raw_speeds,
                "o-",
                alpha=0.3,
                label="Raw (unsmoothed)",
                color="gray",
                markersize=2,
                linewidth=1,
            )

        # Plot smoothed speeds for this method
        ax1.plot(
            frames,
            smoothed_speeds,
            "o-",
            label=method_name,
            color=color,
            markersize=3,
            linewidth=2,
        )

        # Calculate frame-to-frame differences
        if len(smoothed_speeds) > 1:
            diffs = [
                abs(smoothed_speeds[i] - smoothed_speeds[i - 1])
                for i in range(1, len(smoothed_speeds))
            ]
            ax2.plot(
                frames[1:],
                diffs,
                "o-",
                label=method_name,
                color=color,
                markersize=3,
                linewidth=2,
            )

    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel("Speed (mph)")
    ax1.set_title(f"Track {track_id} - Comparison of Smoothing Methods")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("Speed Change (mph)")
    ax2.set_title(f"Track {track_id} - Frame-to-Frame Speed Changes")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Comparison plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize speed data from debug JSONL files"
    )
    parser.add_argument("debug_jsonl", type=str, help="Path to debug JSONL file")
    parser.add_argument(
        "--track-id", type=int, required=True, help="Track ID to visualize"
    )
    parser.add_argument(
        "--output", type=str, help="Output file path for plot (e.g., plot.png)"
    )
    parser.add_argument(
        "--bbox-plot", action="store_true", help="Show bbox size variations vs speed"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare with other debug files (format: name:path name:path)",
    )

    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is not installed")
        print("Install with: pip install matplotlib")
        return 1

    debug_path = Path(args.debug_jsonl)
    if not debug_path.exists():
        print(f"Error: Debug file not found: {debug_path}")
        return 1

    # Read data
    print(f"Reading: {debug_path}")
    data = read_debug_jsonl(debug_path)

    # Filter for track
    track_data = [d for d in data if d["track_id"] == args.track_id]
    if not track_data:
        print(f"Error: Track ID {args.track_id} not found")
        return 1

    print(f"Found {len(track_data)} records for track {args.track_id}")

    # Generate appropriate plot
    if args.compare:
        # Comparison plot
        debug_files = {"Current": debug_path}
        for item in args.compare:
            if ":" in item:
                name, path = item.split(":", 1)
                debug_files[name] = Path(path)

        plot_comparison_multiple_methods(debug_files, args.track_id, args.output)

    elif args.bbox_plot:
        # Bbox vs speed plot
        plot_bbox_size_vs_speed(track_data, args.track_id, args.output)

    else:
        # Standard speed plot
        plot_track_speeds(track_data, args.track_id, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
