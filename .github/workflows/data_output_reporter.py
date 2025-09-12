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

    for v in vars1 & vars2:
        da1 = ds1[v]
        da2 = ds2[v]

        diffs = []
        if da1.shape != da2.shape:
            diffs.append(f"Shape mismatch {da1.shape} vs {da2.shape}")

        for c in da1.coords:
            if c not in da2.coords:
                diffs.append(f"Coordinate '{c}' missing in ds2")
            elif not np.allclose(da1[c], da2[c], rtol=rtol, atol=atol, equal_nan=True):
                diffs.append(f"Coordinate '{c}' values differ")

        for c in da2.coords:
            if c not in da1.coords:
                diffs.append(f"Coordinate '{c}' missing in ds1")

        try:
            if not np.allclose(da1.values, da2.values, rtol=rtol, atol=atol, equal_nan=True):
                diffs.append("Data values differ")
        except Exception as e:
            diffs.append(f"Could not compare values: {e!r}")

        if da1.attrs != da2.attrs:
            diffs.append(f"Attribute mismatch: {da1.attrs} vs {da2.attrs}")

        if diffs:
            report["variable_diffs"][v] = diffs

    if ds1.attrs != ds2.attrs:
        report["attr_diffs"] = {"ds1": ds1.attrs, "ds2": ds2.attrs}

    for c in set(ds1.coords) | set(ds2.coords):
        if c not in ds1.coords:
            report["coord_diffs"][c] = "Missing in ds1"
        elif c not in ds2.coords:
            report["coord_diffs"][c] = "Missing in ds2"
        else:
            if not np.allclose(ds1[c], ds2[c], rtol=rtol, atol=atol, equal_nan=True):
                report["coord_diffs"][c] = "Values differ"
            elif ds1[c].attrs != ds2[c].attrs:
                report["coord_diffs"][c] = f"Attr mismatch: {ds1[c].attrs} vs {ds2[c].attrs}"

    return report


def format_report_md(report: Dict[str, Any]) -> str:
    """Format report dictionary into markdown string."""
    lines = ["# Dataset Comparison Report"]

    if not any(report.values()):
        return "# ✅ Datasets match perfectly!"

    if report["missing_in_ds1"]:
        lines.append("## Variables missing in PR dataset")
        lines.append("| Variable |")
        lines.append("|----------|")
        for v in report["missing_in_ds1"]:
            lines.append(f"| {v} |")

    if report["missing_in_ds2"]:
        lines.append("## Variables missing in Main dataset")
        lines.append("| Variable |")
        lines.append("|----------|")
        for v in report["missing_in_ds2"]:
            lines.append(f"| {v} |")

    if report["variable_diffs"]:
        lines.append("## Variable differences")
        lines.append("| Variable | Issue |")
        lines.append("|----------|-------|")
        for v, diffs in report["variable_diffs"].items():
            for d in diffs:
                lines.append(f"| {v} | {d} |")

    if report["attr_diffs"]:
        lines.append("## Dataset attribute differences")
        lines.append("```")
        lines.append(str(report["attr_diffs"]))
        lines.append("```")

    if report["coord_diffs"]:
        lines.append("## Coordinate differences")
        lines.append("| Coordinate | Issue |")
        lines.append("|------------|-------|")
        for c, d in report["coord_diffs"].items():
            lines.append(f"| {c} | {d} |")

    return "\n".join(lines)


def main():
    args = parse_arguments()
    ds_original = xr.open_dataset(args.orgfile)
    ds_new = xr.open_dataset(args.newfile)

    report = compare_datasets(ds_original, ds_new)
    markdown = format_report_md(report)

    with open("report.md", "w") as f:
        f.write(markdown)


def parse_arguments():
    parser = ArgumentParser(description="Data output comparison report generator")
    parser.add_argument('-o', '--orgfile', type=str, required=True,
                        help='Path to original file to compare to')
    parser.add_argument('-n', '--newfile', type=str, required=True,
                        help='Path to new file to compare against')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
