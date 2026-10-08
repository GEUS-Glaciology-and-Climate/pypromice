import json
import statistics
import xarray as xr
import numpy as np
from argparse import ArgumentParser
from typing import Dict, Any

def compare_datasets(ds1: xr.Dataset,
                     ds2: xr.Dataset,
                     rtol=1e-6,
                     atol=1e-12
) -> dict:
    report = {
        "missing_in_ds1": [],
        "missing_in_ds2": [],
        "variable_diffs": {},
        "attr_diffs": {},
        "coord_diffs": {}
    }

    vars1 = set(ds1.data_vars)
    vars2 = set(ds2.data_vars)
    report["missing_in_ds1"] = sorted(vars2 - vars1)
    report["missing_in_ds2"] = sorted(vars1 - vars2)

    # Compare shared variables
    for v in vars1 & vars2:
        da1 = ds1[v]
        da2 = ds2[v]
        diffs = []

        # Shape comparison for variable data
        if da1.shape != da2.shape:
            diffs.append(f"Data shape mismatch {da1.shape} vs {da2.shape}")

        # Compare variable coordinates
        all_coords = set(da1.coords) | set(da2.coords)
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

            # Shape check first
            if a.shape != b.shape:
                diffs.append(f"Coordinate '{c}' shape mismatch {a.shape} vs {b.shape}")
            else:
                if np.issubdtype(a.dtype, np.number):
                    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                        diffs.append(f"Coordinate '{c}' values differ")
                else:
                    if not np.array_equal(a, b):
                        diffs.append(f"Coordinate '{c}' values differ")

            # Compare coordinate attributes
            if da1[c].attrs != da2[c].attrs:
                diffs.append(f"Coordinate '{c}' attribute mismatch: {da1[c].attrs} vs {da2[c].attrs}")

        # Compare variable data if shapes match
        if da1.shape == da2.shape:
            if np.issubdtype(da1.values.dtype, np.number):
                if not np.allclose(da1.values, da2.values, rtol=rtol, atol=atol, equal_nan=True):
                    diffs.append("Data values differ")
            else:
                if not np.array_equal(da1.values, da2.values):
                    diffs.append("Data values differ")

        # Compare variable attributes
        if da1.attrs != da2.attrs:
            diffs.append(f"Attribute mismatch: {da1.attrs} vs {da2.attrs}")

        if diffs:
            report["variable_diffs"][v] = diffs

    # Compare dataset attributes
    if sorted(ds1.attrs) != sorted(ds2.attrs):
        report["attr_diffs"] = {"ds1": sorted(ds1.attrs), "ds2": sorted(ds2.attrs)}

    # Compare dataset-level coordinates
    all_ds_coords = set(ds1.coords) | set(ds2.coords)
    for c in all_ds_coords:
        in_ds1 = c in ds1.coords
        in_ds2 = c in ds2.coords

        if not in_ds1:
            report["coord_diffs"][c] = "Missing in ds1"
            continue
        if not in_ds2:
            report["coord_diffs"][c] = "Missing in ds2"
            continue

        a = ds1[c].values
        b = ds2[c].values

        # Shape check first
        if a.shape != b.shape:
            report["coord_diffs"][c] = f"Shape mismatch {a.shape} vs {b.shape}"
        else:
            if np.issubdtype(a.dtype, np.number):
                if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                    report["coord_diffs"][c] = "Values differ"
            else:
                if not np.array_equal(a, b):
                    report["coord_diffs"][c] = "Values differ"

            # Compare coordinate attributes
            if ds1[c].attrs != ds2[c].attrs:
                report["coord_diffs"][c] = f"Attr mismatch: {ds1[c].attrs} vs {ds2[c].attrs}"

    return report

def format_report_md(report: Dict[str, Any]) -> str:
    """Format report dictionary into markdown string."""
    lines = ["# Dataset Comparison Report"]

    if not any(report.values()):
        lines.append("## ✅ Datasets match perfectly!")
        lines.append("No differences have been found between " +
                     "the datasets produced using the PR " +
                     "branch and the main branch.")
        lines.append("⚠️ This report is generated from a small " +
                     "subset of test data that does not reflect " +
                     "all scenarios. Please check using more " +
                     "input data if you suspect changes to the " +
                     "output data have occurred.")
        return "\n".join(lines)

    lines.append("Differences have been found between " +
                 "the datasets produced using the PR " +
                 "branch and the main branch.")
    lines.append("If you did not expect changes to be "+
                 "made to the dataset from your PR " +
                 "then please check this report and" +
                 "update your branch accordingly.")
  
    lines.append("## Variables missing in PR dataset")
    if report["missing_in_ds1"]:
      lines.append("| Variable |")
      lines.append("|----------|")
      for v in report["missing_in_ds1"]:
            lines.append(f"| {v} |")
    else:
      lines.append("None")

    lines.append("## Variables missing in Main dataset")
    if report["missing_in_ds2"]:
      lines.append("| Variable |")
      lines.append("|----------|")
      for v in report["missing_in_ds2"]:
            lines.append(f"| {v} |")
    else:
      lines.append("None")

    lines.append("## Variable differences")
    if report["variable_diffs"]:
        lines.append("| Variable | Issue |")
        lines.append("|----------|-------|")
        for v, diffs in report["variable_diffs"].items():
            for d in diffs:
                lines.append(f"| {v} | {d} |")
    else:
      lines.append("None")

    lines.append("## Dataset attribute differences")
    if report["attr_diffs"]:
        lines.append("Original dataset attributes (main branch)")
        lines.append("```")
        lines.append(str(report["attr_diffs"]["ds1"]))
        lines.append("```")
        lines.append("New dataset atrributes (PR branch)")
        lines.append("```")
        lines.append(str(report["attr_diffs"]["ds2"]))
        lines.append("```")
    else:
      lines.append("None")

    lines.append("## Coordinate differences")
    if report["coord_diffs"]:
        lines.append("| Coordinate | Issue |")
        lines.append("|------------|-------|")
        for c, d in report["coord_diffs"].items():
            lines.append(f"| {c} | {d} |")
    else:
      lines.append("None")

    return "\n".join(lines)


# Relative runtime change below which a difference is treated as noise
RUNTIME_NOISE = 0.10


def format_timing_md(timings_org: Dict[str, list], timings_new: Dict[str, list]) -> str:
    """Format the runtime comparison (median of the timed runs) as markdown."""
    lines = ["# Runtime Comparison", ""]
    lines.append("| Step | Main (s) | PR (s) | Change | |")
    lines.append("|------|----------|--------|--------|---|")

    total_org = total_new = 0.0
    for step in timings_org:
        if step not in timings_new:
            continue
        org = statistics.median(timings_org[step])
        new = statistics.median(timings_new[step])
        total_org += org
        total_new += new
        lines.append(_timing_row(step, org, new))
    lines.append(_timing_row("**total**", total_org, total_new))

    n_runs = min(len(v) for v in timings_new.values())
    lines.append("")
    lines.append(f"Median of {n_runs} timed runs per step, after one warm-up "
                 "run, both branches on the same runner. "
                 f"Differences below {RUNTIME_NOISE:.0%} are treated as noise. "
                 "⚠️ This is the small test dataset: it is dominated by "
                 "start-up and I/O costs and does not reflect the runtime of a "
                 "full-size station.")
    lines.append("")
    lines.append("<details><summary>Individual runs (s)</summary>")
    lines.append("")
    lines.append("| Step | Branch | Runs |")
    lines.append("|------|--------|------|")
    for step in timings_org:
        for label, timings in (("main", timings_org), ("PR", timings_new)):
            runs = ", ".join(f"{t:.1f}" for t in timings.get(step, []))
            lines.append(f"| {step} | {label} | {runs} |")
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def _timing_row(step: str, org: float, new: float) -> str:
    change = (new - org) / org if org > 0 else 0.0
    if change <= -RUNTIME_NOISE:
        verdict = f"🚀 {org / new:.2f}x faster"
    elif change >= RUNTIME_NOISE:
        verdict = f"🐢 {new / org:.2f}x slower"
    else:
        verdict = "≈ no significant change"
    return f"| {step} | {org:.1f} | {new:.1f} | {change:+.1%} | {verdict} |"


def main():
    args = parse_arguments()
    ds_original = xr.open_dataset(args.orgfile)
    ds_new = xr.open_dataset(args.newfile)

    report = compare_datasets(ds_original, ds_new)
    markdown = format_report_md(report)

    if args.timings_org and args.timings_new:
        with open(args.timings_org) as f:
            timings_org = json.load(f)
        with open(args.timings_new) as f:
            timings_new = json.load(f)
        markdown += "\n\n" + format_timing_md(timings_org, timings_new)

    with open("report.md", "w") as f:
        f.write(markdown)


def parse_arguments():
    parser = ArgumentParser(description="Data output comparison report generator")
    parser.add_argument('-o', '--orgfile', type=str, required=True,
                        help='Path to original file to compare to')
    parser.add_argument('-n', '--newfile', type=str, required=True,
                        help='Path to new file to compare against')
    parser.add_argument('--timings_org', type=str, default=None,
                        help='Runtimes json of the original (main) branch')
    parser.add_argument('--timings_new', type=str, default=None,
                        help='Runtimes json of the new (PR) branch')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
