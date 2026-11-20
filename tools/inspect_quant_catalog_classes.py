#!/usr/bin/env python3
"""Inspect and instantiate classes listed in quantization_block_analysis.json."""

import importlib
import inspect
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generators.registry import ModelRegistry

def main():
    json_path = REPO_ROOT / "quantization_block_analysis.json"
    with open(json_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    print(f"Total catalog entries: {len(catalog)}")
    
    # Also discover all specs from ModelRegistry
    specs = ModelRegistry.discover(REPO_ROOT / "generators")
    spec_map = {s.target.__name__ if inspect.isclass(s.target) else s.name: s for s in specs}
    
    instantiable = 0
    failed = []
    
    for cname, entry in catalog.items():
        src_file = entry.get("source_file", "")
        # Try to find class
        cls_obj = None
        
        # Check if in spec_map
        if cname in spec_map:
            try:
                m = spec_map[cname].instantiate()
                instantiable += 1
                continue
            except Exception as e:
                pass
                
        # Try importing from generators.<module>
        mod_name = Path(src_file).stem
        try:
            mod = importlib.import_module(f"generators.{mod_name}")
            cls_obj = getattr(mod, cname, None)
        except Exception as e:
            try:
                mod = importlib.import_module(f"generators.addons.{mod_name}")
                cls_obj = getattr(mod, cname, None)
            except Exception as e2:
                cls_obj = None
                
        if cls_obj is not None and inspect.isclass(cls_obj) and issubclass(cls_obj, nn.Module):
            # Try instantiating with default args or common args
            instance = None
            for init_args in [
                (),
                (32,),
                (32, 64),
                (32, 4, 64),
            ]:
                try:
                    instance = cls_obj(*init_args)
                    break
                except Exception:
                    continue
            if instance is not None:
                instantiable += 1
            else:
                failed.append((cname, src_file, "constructor failed with standard args"))
        else:
            failed.append((cname, src_file, "class not found or not nn.Module"))

    print(f"Instantiable classes: {instantiable} / {len(catalog)}")
    if failed:
        print(f"Failed count: {len(failed)}")
        for f_item in failed[:10]:
            print(" ", f_item)

if __name__ == "__main__":
    main()
