#!/usr/bin/env python3
"""CLI Entrypoint for the Unified DL Architecture Zoo Generators."""

from generators.registry import ModelRegistry

if __name__ == "__main__":
    ModelRegistry.cli()
