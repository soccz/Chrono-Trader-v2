#!/usr/bin/env python3
import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.output_contract import backfill_output_manifests


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill missing output contract manifests from existing artifacts.")
    ap.add_argument("--modes", nargs="*", default=["intraday", "morning", "refresh-db"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    written = backfill_output_manifests(args.modes, overwrite=bool(args.overwrite))
    for mode in args.modes:
        print({"mode": mode, "manifest_path": written.get(mode)})


if __name__ == "__main__":
    main()
