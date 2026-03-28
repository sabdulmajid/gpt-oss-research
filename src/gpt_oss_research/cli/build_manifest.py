from __future__ import annotations

import argparse
import json

from ..io import write_json
from ..manifests import build_frozen_manifest, load_manifest_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen dataset manifest from a YAML spec.")
    parser.add_argument("spec_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    spec = load_manifest_spec(args.spec_path)
    manifest = build_frozen_manifest(spec, source_path=args.spec_path)
    write_json(args.output_path, manifest)
    print(json.dumps({"output_path": args.output_path, "manifest_name": manifest["manifest_name"]}, indent=2))


if __name__ == "__main__":
    main()

