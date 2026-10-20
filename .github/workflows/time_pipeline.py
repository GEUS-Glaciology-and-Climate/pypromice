"""Run the L0 -> L3 test processing several times and record the runtimes.

Only uses command line options that exist on both main and PR branches, so
the same script can time both. One untimed warm-up run is done first (file
cache, imports), then ``--repeats`` timed runs of each step. The runtimes
(seconds) are written to ``<outdir>/timings.json``.
"""
import json
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path


def run(cmd):
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - start


def main():
    parser = ArgumentParser(description="Time the L0 to L3 test processing")
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("-d", "--data_issues_path", required=True)
    parser.add_argument("-r", "--repeats", type=int, default=3)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.data_issues_path).mkdir(parents=True, exist_ok=True)

    steps = {
        "get_l2": [
            "get_l2", "-c", "tests/data/test_config1_raw.toml",
            "-i", "tests/data/", "-o", f"{outdir}/",
            "--data_issues_path", args.data_issues_path,
            "--declination_path",
            "tests/data/magnetic_declination_configurations/test_magdec_config1.toml",
            "--write_60min",
        ],
        "get_l2tol3": [
            "get_l2tol3", "-c", "tests/data/station_configurations/TEST1.toml",
            "-i", f"{outdir}/TEST1/TEST1_mixed.nc", "-o", f"{outdir}/",
            "--data_issues_path", args.data_issues_path,
            "--write_60min",
        ],
    }

    timings = {name: [] for name in steps}
    for repeat in range(args.repeats + 1):  # first run is the warm-up
        for name, cmd in steps.items():
            elapsed = run(cmd)
            if repeat > 0:
                timings[name].append(elapsed)
            print(f"{name} run {repeat}{' (warm-up)' if repeat == 0 else ''}: {elapsed:.1f} s")

    (outdir / "timings.json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
