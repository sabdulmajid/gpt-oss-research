from __future__ import annotations

import argparse
import json
import sys

from ..manifests import load_manifest_spec, validate_manifest_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a manifest spec or frozen manifest.")
    parser.add_argument("manifest_path")
    parser.add_argument("--frozen", action="store_true")
    args = parser.parse_args()

    spec = load_manifest_spec(args.manifest_path, frozen=args.frozen)
    validation = validate_manifest_spec(spec, frozen=args.frozen)
    payload = {"ok": validation.ok, "errors": validation.errors, "warnings": validation.warnings}
    print(json.dumps(payload, indent=2))
    if not validation.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

