import glob, os
import xarray as xr
from pathlib import Path
from pypromice.utilities import compare

rtol = 1e-6
atol = 1e-12

dir1 = "/data/aws-ops/aws-l3-v1.6.0/sites"
dir2 = "/data/aws-ops/aws-l3-v1.7.0/sites"
outdir1 = "/home/pho/Desktop/out/"

site_names = [Path(s).name for s in glob.glob(f"{dir1}/*")]
# site = "NUK_U"

for site in site_names:

    out = f"{outdir1}/{site}/"
    os.makedirs(out, exist_ok=True)

    # Load datasets
    ds1 = xr.open_dataset(f"{dir1}/{site}/{site}_hour.nc")
    ds2 = xr.open_dataset(f"{dir2}/{site}/{site}_hour.nc")

    # Compare datasets
    report = compare.compare_datasets(ds1, ds2, rtol=rtol, atol=atol)

    # Generate and save Markdown report
    markdown = compare.format_report_md(report)
    report_file = f"{out}/{site}_hour.md"
    with open(report_file, "w") as f:
        f.write(markdown)
    print(f"\nMarkdown report saved to {report_file}")

    # Optional: Plot variable differences
    compare.plot_variable_differences(ds1, ds2, report, out)
