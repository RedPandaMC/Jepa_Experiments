#!/usr/bin/env python
r"""Download and merge Jena Climate 2009–2016 data.

Downloads ZIP files from the Max Planck Institute for Biogeochemistry
weather station, extracts the CSVs, and merges them into a single
``data/jena_climate_2009_2016.csv`` file.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://www.bgc-jena.mpg.de/wetter"
# Files from 2009a through 2016b (16 semesterly ZIPs)
YEARS = range(2009, 2017)
HALVES = ["a", "b"]

OUTPUT_DIR = Path("data")
OUTPUT_FILE = "jena_climate_2009_2016.csv"


def download_and_extract(url: str) -> str | None:
    """Download a ZIP file and return the first CSV content as text."""
    print(f"  Downloading {url} ...")
    try:
        resp = urlopen(url, timeout=60)
        data = resp.read()
    except Exception as e:
        print(f"  Warning: failed to download {url}: {e}")
        return None

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        print(f"  Warning: bad ZIP from {url}")
        return None

    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
    if not csv_names:
        print(f"  Warning: no CSV found in {url}")
        return None

    return zf.read(csv_names[0]).decode("ascii", errors="replace")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_FILE

    all_lines: list[str] = []
    header_written = False

    for year in YEARS:
        for half in HALVES:
            fname = f"mpi_roof_{year}{half}.zip"
            url = f"{BASE_URL}/{fname}"
            csv_content = download_and_extract(url)
            if csv_content is None:
                continue

            lines = csv_content.strip().split("\n")
            if not header_written:
                all_lines.append(lines[0])
                header_written = True
            for line in lines[1:]:
                if line.strip():
                    all_lines.append(line)

    out_path.write_text("\n".join(all_lines) + "\n")
    print(f"Merged {len(all_lines) - 1} rows → {out_path}")


if __name__ == "__main__":
    main()
