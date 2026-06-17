"""GPU — 3-way contrast probe (S-U, S-O, U-O pairs).

Collects three contrast pairs per example + direct A/B/C logits.

Example:
    uv run --env-file .env python scripts/collect_contrast_probe_threeway.py \\
        --category Gender_identity --max-examples 500
"""
from __future__ import annotations

import argparse

from mech_interp_bbq.contrast_collect import add_shared_collect_args, run_collect


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect 3-way contrast-probe activations.")
    add_shared_collect_args(ap)
    run_collect(ap.parse_args(), mode="threeway")


if __name__ == "__main__":
    main()
