import os, json
import numpy as np
import xarray as xr
from typing import Dict, Any
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def compare_datasets(ds1: xr.Dataset,
                     ds2: xr.Dataset,
                     rtol: float = 1e-6,
                     atol: float = 1e-12) -> dict:
    """
    Compare two xarray datasets and return a report dictionary.
    Also prints summaries to console.
    """
    report = {
        "missing_in_ds1": [],
        "missing_in_ds2": [],
        "variable_diffs": {},
        "attr_diffs": {},
        "coord_diffs": {}
    }

    # Alphabetical variable lists
    vars1 = sorted(ds1.data_vars)
    vars2 = sorted(ds2.data_vars)

    # Missing variables
    report["missing_in_ds1"] = sorted(set(vars2) - set(vars1))
    report["missing_in_ds2"] = sorted(set(vars1) - set(vars2))

    # Print missing variables
    if report["missing_in_ds1"]:
        print("\nVariables missing in ds1:")
        for v in report["missing_in_ds1"]:
            print(f" - {v}")
    if report["missing_in_ds2"]:
        print("\nVariables missing in ds2:")
        for v in report["missing_in_ds2"]:
            print(f" - {v}")

    # Shared variables
    shared_vars = sorted(set(vars1) & set(vars2))

    # Compare shared variables
    for v in shared_vars:
        da1 = ds1[v]
        da2 = ds2[v]
        diffs = []
        stats = {}

        # Shape comparison
        if da1.shape != da2.shape:
            diffs.append(f"Data shape mismatch {da1.shape} vs {da2.shape}")

        # Coordinate comparison
        all_coords = sorted(set(da1.coords) | set(da2.coords))
        for c in all_coords:
            in_ds1 = c in da1.coords
            in_ds2 = c in da2.coords
            if not in_ds1:
                diffs.append(f"Coordinate '{c}' missing in ds1")
                continue
            if not in_ds2:
                diffs.append(f"Coordinate '{c}' missing in ds2")
                continue
            a = da1[c].values
            b = da2[c].values
            if a.shape != b.shape:
                diffs.append(f"Coordinate '{c}' shape mismatch {a.shape} vs {b.shape}")
            else:
                if np.issubdtype(a.dtype, np.number):
                    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                        diffs.append(f"Coordinate '{c}' values differ")
                else:
                    if not np.array_equal(a, b):
                        diffs.append(f"Coordinate '{c}' values differ")
            if da1[c].attrs != da2[c].attrs:
                diffs.append(f"Coordinate '{c}' attribute mismatch: {da1[c].attrs} vs {da2[c].attrs}")

        # Variable data comparison
        if da1.shape == da2.shape:
            if np.issubdtype(da1.values.dtype, np.number):
                if not np.allclose(da1.values, da2.values, rtol=rtol, atol=atol, equal_nan=True):
                    diffs.append("Data values differ")
                    da1a, da2a = xr.align(da1, da2, join='inner')
                    abs_diff = np.abs(da1a - da2a)
                    tolerance = atol + rtol * np.abs(da2a)
                    abs_err = abs_diff - tolerance
                    abs_err = abs_err.where(abs_err > 0, 0)
                    mean_abs_err = float(abs_err.mean(skipna=True))
                    max_abs_err = float(abs_err.max(skipna=True))
                    std_abs_err = float(abs_err.std(skipna=True))
                    da2_mean = float(da2a.mean(skipna=True))
                    mean_pct_err = float((abs_err / np.abs(da2_mean) * 100).mean(skipna=True)) if da2_mean != 0 else float('nan')
                    stats = {
                        "mean_abs_err": round(mean_abs_err, 3),
                        "max_abs_err": round(max_abs_err, 3),
                        "std_abs_err": round(std_abs_err, 3),
                        "mean_pct_err": round(mean_pct_err, 3)
                    }
                    if all(v == 0.0 or np.isnan(v) for v in stats.values()):
                        stats["note"] = "< differences within tolerance or below rounding precision >"
            else:
                if not np.array_equal(da1.values, da2.values):
                    diffs.append("Data values differ (non-numeric)")
        else:
            diffs.append("Data shapes differ, cannot compute statistics")

        # Attribute comparison
        if da1.attrs != da2.attrs:
            diffs.append(f"Attribute mismatch: {da1.attrs} vs {da2.attrs}")

        # Store results
        if diffs or stats:
            report["variable_diffs"][v] = {"diffs": diffs, "stats": stats if stats else None}

    # Dataset attribute differences
    if sorted(ds1.attrs) != sorted(ds2.attrs):
        report["attr_diffs"] = {"ds1": sorted(ds1.attrs), "ds2": sorted(ds2.attrs)}
        print("\n--- Dataset Attribute Differences ---")
        print("ds1:", report["attr_diffs"]["ds1"])
        print("ds2:", report["attr_diffs"]["ds2"])

    # Coordinate differences
    all_ds_coords = sorted(set(ds1.coords) | set(ds2.coords))
    for c in all_ds_coords:
        in_ds1 = c in ds1.coords
        in_ds2 = c in ds2.coords
        if not in_ds1:
            report["coord_diffs"][c] = "Missing in ds1"
        elif not in_ds2:
            report["coord_diffs"][c] = "Missing in ds2"
        else:
            a = ds1[c].values
            b = ds2[c].values
            if a.shape != b.shape:
                report["coord_diffs"][c] = f"Shape mismatch {a.shape} vs {b.shape}"
            else:
                if np.issubdtype(a.dtype, np.number):
                    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                        report["coord_diffs"][c] = "Values differ"
                else:
                    if not np.array_equal(a, b):
                        report["coord_diffs"][c] = "Values differ"
                if ds1[c].attrs != ds2[c].attrs:
                    report["coord_diffs"][c] = f"Attr mismatch: {ds1[c].attrs} vs {ds2[c].attrs}"
        if c in report["coord_diffs"]:
            print(f"Coordinate '{c}' difference: {report['coord_diffs'][c]}")

    # Print variable differences with stats
    if report["variable_diffs"]:
        print("\n--- Variable Differences ---")
        for var in sorted(report["variable_diffs"].keys()):
            diffinfo = report["variable_diffs"][var]
            print(f"\nVariable: {var}")
            for d in diffinfo["diffs"]:
                print(f" - {d}")
            if diffinfo.get("stats"):
                print("   Stats:", diffinfo["stats"])

    return report


def format_report_md(report: Dict[str, Any]) -> str:
    """Generate Markdown report from report dict."""
    lines = ["# Dataset Comparison Report"]

    # Missing variables
    lines.append("## Variables missing in ds1")
    if report["missing_in_ds1"]:
        lines.append("| Variable |")
        lines.append("|----------|")
        for v in report["missing_in_ds1"]:
            lines.append(f"| {v} |")
    else:
        lines.append("None")

    lines.append("## Variables missing in ds2")
    if report["missing_in_ds2"]:
        lines.append("| Variable |")
        lines.append("|----------|")
        for v in report["missing_in_ds2"]:
            lines.append(f"| {v} |")
    else:
        lines.append("None")

    # Variable differences
    lines.append("## Variable differences")
    if report["variable_diffs"]:
        lines.append("| Variable | Issue | mean_abs_err | max_abs_err | std_abs_err | mean_pct_err |")
        lines.append("|----------|-------|--------------|------------|------------|--------------|")
        for var in sorted(report["variable_diffs"].keys()):
            diffinfo = report["variable_diffs"][var]
            stats = diffinfo.get("stats", {})
            mean_abs = stats.get("mean_abs_err", "")
            max_abs = stats.get("max_abs_err", "")
            std_abs = stats.get("std_abs_err", "")
            mean_pct = stats.get("mean_pct_err", "")
            for d in diffinfo["diffs"]:
                lines.append(f"| {var} | {d} | {mean_abs} | {max_abs} | {std_abs} | {mean_pct} |")
    else:
        lines.append("None")

    # Dataset attributes
    lines.append("## Dataset attribute differences")
    if report["attr_diffs"]:
        lines.append("ds1 attributes:")
        lines.append("```")
        lines.append(str(report["attr_diffs"]["ds1"]))
        lines.append("```")
        lines.append("ds2 attributes:")
        lines.append("```")
        lines.append(str(report["attr_diffs"]["ds2"]))
        lines.append("```")
    else:
        lines.append("None")

    # Coordinate differences
    lines.append("## Coordinate differences")
    if report["coord_diffs"]:
        lines.append("| Coordinate | Issue |")
        lines.append("|------------|-------|")
        for c, d in report["coord_diffs"].items():
            lines.append(f"| {c} | {d} |")
    else:
        lines.append("None")

    return "\n".join(lines)

def plot_variable_differences(ds1: xr.Dataset,
                              ds2: xr.Dataset,
                              report: dict,
                              outdir: str):
    """
    Plot variables that differ between two datasets.

    Args:
        ds1, ds2: xarray Datasets to compare
        report: comparison report from compare_datasets()
        outdir: directory to save plots
    """
    # Parse version from dataset attributes
    try:
        v1 = json.loads(ds1.attrs.get("source", '{}')).get("pypromice", "unknown")
    except Exception:
        v1 = "unknown"
    try:
        v2 = json.loads(ds2.attrs.get("source", '{}')).get("pypromice", "unknown")
    except Exception:
        v2 = "unknown"

    for var in sorted(report["variable_diffs"].keys()):
        diffinfo = report["variable_diffs"][var]
        if not diffinfo.get("stats"):
            continue  # skip if no stats to plot
        if diffinfo["stats"].get("mean_pct_err", 0) < 0.005:
            continue  # skip if differences are negligible

        da1 = ds1[var]
        da2 = ds2[var]
        da1a, da2a = xr.align(da1, da2, join='inner')

        try:
            diff_da = da1a - da2a
            plt.figure(figsize=(10, 5))
            if diff_da.ndim == 1:
                plt.plot(da1a, label=f"v{v1}")
                plt.plot(da2a, label=f"v{v2}")
                plt.plot(diff_da, label="Difference", linestyle='--')
                plt.title(f"Variable: {var}")
                plt.legend()
            elif diff_da.ndim == 2:
                plt.subplot(1, 3, 1)
                da1a.plot()
                plt.title(f"{var} (v{v1})")
                plt.subplot(1, 3, 2)
                da2a.plot()
                plt.title(f"{var} (v{v2})")
                plt.subplot(1, 3, 3)
                diff_da.plot()
                plt.title(f"{var} Difference")
                plt.tight_layout()
            else:
                print(f"Skipping plot for {var}: unsupported dims ({diff_da.ndim})")

            # Save the plot
            plt.savefig(f"{outdir}/{var}_diff.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Plotting failed for {var}: {e}")


def parse_arguments():
    parser = ArgumentParser(description="Data output comparison report generator")
    parser.add_argument('-o', '--orgfile', type=str, required=True,
                        help='Path to original file to compare to')
    parser.add_argument('-n', '--newfile', type=str, required=True,
                        help='Path to new file to compare against')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Flag for plotting')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    print(f"Loading {args.orgfile}...")
    ds_original = xr.open_dataset(args.orgfile)

    print(f"Loading {args.newfile}...")
    ds_new = xr.open_dataset(args.newfile)

    print("Generating comparison report...")
    report = compare_datasets(ds_original, ds_new)
    markdown = format_report_md(report)

    with open("report.md", "w") as f:
        print("Writing markdown file to report.md...")
        f.write(markdown)

    if args.plot:
        print("Plotting variables...")
        plot_variable_differences(ds_original, ds_new, report, ".")


if __name__ == "__main__":
    main()
