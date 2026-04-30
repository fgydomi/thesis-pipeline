from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/tables")

METRICS_PATHS = {
    "Cellpose": Path("outputs/metrics/cellpose/test/cellpose_metrics.csv"),
    "StarDist": Path("outputs/metrics/stardist/test/stardist_metrics.csv"),
    "U-Net + Watershed": Path(
        "outputs/metrics/unet_watershed/test/unet_watershed_metrics.csv"
    ),
}

RUNTIME_PATHS = {
    "Cellpose": Path(
        "outputs/predictions/cellpose_manifest/test/cellpose_runtime.csv"
    ),
    "StarDist": Path(
        "outputs/predictions/stardist_manifest/test/stardist_runtime.csv"
    ),
    "U-Net + Watershed": Path(
        "outputs/predictions/unet_watershed_manifest/test/unet_watershed_runtime.csv"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build thesis-ready summary tables from saved metrics and runtime CSV files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where summary tables will be written.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    return pd.read_csv(path)


def build_quantitative_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model_name, csv_path in METRICS_PATHS.items():
        df = load_csv(csv_path)

        successful = df[df["status"] == "success"].copy()
        if successful.empty:
            raise ValueError(f"No successful rows found in metrics CSV: {csv_path}")

        successful["dice_fg"] = successful["dice_fg"].astype(float)
        successful["iou_fg"] = successful["iou_fg"].astype(float)
        successful["aji"] = successful["aji"].astype(float)

        rows.append(
            {
                "model": model_name,
                "mean_dice": successful["dice_fg"].mean(),
                "mean_iou": successful["iou_fg"].mean(),
                "mean_aji": successful["aji"].mean(),
                "n_images": len(successful),
            }
        )

    summary = pd.DataFrame(rows)

    # Rank primarily by AJI, then Dice, then IoU
    summary = summary.sort_values(
        by=["mean_aji", "mean_dice", "mean_iou"],
        ascending=False,
    ).reset_index(drop=True)
    summary["overall_rank"] = range(1, len(summary) + 1)

    # Reorder columns
    summary = summary[
        ["model", "mean_dice", "mean_iou", "mean_aji", "overall_rank", "n_images"]
    ]

    return summary


def build_runtime_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model_name, csv_path in RUNTIME_PATHS.items():
        df = load_csv(csv_path)

        successful = df[df["status"] == "success"].copy()
        if successful.empty:
            raise ValueError(f"No successful rows found in runtime CSV: {csv_path}")

        successful["elapsed_seconds"] = successful["elapsed_seconds"].astype(float)

        rows.append(
            {
                "model": model_name,
                "mean_runtime_s": successful["elapsed_seconds"].mean(),
                "min_runtime_s": successful["elapsed_seconds"].min(),
                "max_runtime_s": successful["elapsed_seconds"].max(),
                "n_images": len(successful),
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(by="mean_runtime_s", ascending=True).reset_index(drop=True)
    summary["speed_rank"] = range(1, len(summary) + 1)

    summary = summary[
        [
            "model",
            "mean_runtime_s",
            "min_runtime_s",
            "max_runtime_s",
            "speed_rank",
            "n_images",
        ]
    ]

    return summary


def format_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()

    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda x: f"{x:.6f}")

    return formatted


def format_for_markdown(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()

    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda x: f"{x:.6f}")

    return formatted


def save_markdown_table(
    df: pd.DataFrame,
    path: Path,
    title: str,
    note: str,
) -> None:
    formatted = format_for_markdown(df)

    with path.open("w", encoding="utf-8") as file:
        file.write(f"{title}\n\n")
        file.write(formatted.to_markdown(index=False))
        file.write("\n\n")
        file.write(f"{note}\n")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    quantitative_df = build_quantitative_summary()
    runtime_df = build_runtime_summary()

    quantitative_csv_path = args.output_dir / "table_quantitative_results.csv"
    runtime_csv_path = args.output_dir / "table_runtime_results.csv"

    quantitative_md_path = args.output_dir / "table_quantitative_results.md"
    runtime_md_path = args.output_dir / "table_runtime_results.md"

    format_for_csv(quantitative_df).to_csv(quantitative_csv_path, index=False)
    format_for_csv(runtime_df).to_csv(runtime_csv_path, index=False)

    save_markdown_table(
        quantitative_df,
        quantitative_md_path,
        title="Table 3. Main quantitative comparison of the three selected model families on the MoNuSeg test split.",
        note=(
            "Note. Dice and IoU were computed on binary foreground masks derived from "
            "the final instance predictions by treating all nonzero labels as nucleus "
            "foreground. AJI was computed on final instance masks. Reported values are "
            "mean per-image results on the official MoNuSeg test split."
        ),
    )

    save_markdown_table(
        runtime_df,
        runtime_md_path,
        title="Table 4. Runtime comparison of the three selected model families on the MoNuSeg test split.",
        note=(
            "Note. Runtime was measured per image at the inference-pipeline level and "
            "does not include metric computation. For the U-Net + Watershed branch, "
            "the reported runtime includes semantic inference followed by watershed-based "
            "instance formation. Reported values are summarized across the official "
            "MoNuSeg test split."
        ),
    )

    print("Summary tables created successfully.")
    print(f"Quantitative CSV: {quantitative_csv_path}")
    print(f"Runtime CSV:      {runtime_csv_path}")
    print(f"Quantitative MD:  {quantitative_md_path}")
    print(f"Runtime MD:       {runtime_md_path}")
    print()

    print("Quantitative summary:")
    print(format_for_markdown(quantitative_df).to_markdown(index=False))
    print()

    print("Runtime summary:")
    print(format_for_markdown(runtime_df).to_markdown(index=False))


if __name__ == "__main__":
    main()