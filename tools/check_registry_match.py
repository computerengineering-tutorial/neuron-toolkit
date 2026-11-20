#!/usr/bin/env python3
"""Check mapping between quantization_block_analysis.json and ModelRegistry specs."""

import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generators.registry import ModelRegistry

def main():
    json_path = REPO_ROOT / "quantization_block_analysis.json"
    with open(json_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    specs = ModelRegistry.discover(REPO_ROOT / "generators")
    
    # Map target class or spec name
    spec_by_cls = {}
    spec_by_name = {}
    for s in specs:
        spec_by_name[s.name] = s
        if inspect.isclass(s.target):
            spec_by_cls[s.target.__name__] = s

    matched = 0
    unmatched = []

    for cname, entry in catalog.items():
        if cname in spec_by_cls:
            matched += 1
        elif cname in spec_by_name:
            matched += 1
        else:
            unmatched.append(cname)

    print(f"Total catalog classes: {len(catalog)}")
    print(f"Matched with ModelRegistry specs: {matched}")
    print(f"Unmatched classes: {len(unmatched)}")
    print("Unmatched sample:", unmatched[:15])

if __name__ == "__main__":
    main()
