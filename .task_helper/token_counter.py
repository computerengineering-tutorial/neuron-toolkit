#!/usr/bin/env python3
"""Token Counter Helper Script
Reads all readable files from a specified folder, concatenates their contents,
and calculates token counts using tiktoken.
"""

import argparse
import os
import sys
from pathlib import Path

import tiktoken


def is_binary_file(file_path: Path) -> bool:
    """Check if a file appears to be binary by checking for null bytes in initial chunk."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return True
        return False
    except Exception:
        return True


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens for a given text string using tiktoken."""
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except ValueError:
        enc = tiktoken.encoding_for_model(encoding_name)
    return len(enc.encode(text, disallowed_special=()))


def scan_and_count(
    folder_path: str,
    encoding_name: str = "cl100k_base",
    show_details: bool = True,
    exclude_hidden: bool = True,
):
    target_dir = Path(folder_path).resolve()
    if not target_dir.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)
    if not target_dir.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        sys.exit(1)

    print(f"Scanning directory: {target_dir}")
    print(f"Base encoding: {encoding_name}\n" + "-" * 60)

    try:
        enc = tiktoken.get_encoding(encoding_name)
    except ValueError:
        enc = tiktoken.encoding_for_model(encoding_name)

    file_stats = []
    combined_texts = []
    skipped_files = []

    for root, dirs, files in os.walk(target_dir):
        if exclude_hidden:
            dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file_name in sorted(files):
            if exclude_hidden and file_name.startswith("."):
                continue

            file_path = Path(root) / file_name
            rel_path = file_path.relative_to(target_dir)

            if is_binary_file(file_path):
                skipped_files.append((str(rel_path), "Binary file"))
                continue

            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    content = f.read()

                tokens = len(enc.encode(content, disallowed_special=()))
                chars = len(content)
                words = len(content.split())

                file_stats.append({
                    "path": str(rel_path),
                    "tokens": tokens,
                    "chars": chars,
                    "words": words,
                })
                combined_texts.append(content)

            except Exception as e:
                skipped_files.append((str(rel_path), f"Error reading: {e}"))

    # Print breakdown per file if requested
    if show_details and file_stats:
        print(f"{'File Path':<40} {'Tokens':>10} {'Words':>10} {'Chars':>10}")
        print("-" * 74)
        for stat in file_stats:
            print(
                f"{stat['path']:<40} {stat['tokens']:>10,d} {stat['words']:>10,d} {stat['chars']:>10,d}"
            )
        print("-" * 74)

    # Combined text stats
    full_combined_text = "\n\n".join(combined_texts)
    combined_tokens = len(enc.encode(full_combined_text, disallowed_special=()))
    total_chars = len(full_combined_text)
    total_words = len(full_combined_text.split())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed files : {len(file_stats)}")
    print(f"Skipped files         : {len(skipped_files)}")
    print(f"Total characters      : {total_chars:,d}")
    print(f"Total words           : {total_words:,d}")
    print(f"Combined token count  : {combined_tokens:,d} ({encoding_name})")

    # Extra comparison with o200k_base (GPT-4o) if available
    for alt_enc_name in ["o200k_base", "p50k_base"]:
        if alt_enc_name != encoding_name:
            try:
                alt_enc = tiktoken.get_encoding(alt_enc_name)
                alt_tokens = len(alt_enc.encode(full_combined_text, disallowed_special=()))
                print(f"Alternative ({alt_enc_name:<11}): {alt_tokens:,d} tokens")
            except Exception:
                pass

    if skipped_files:
        print("\nSkipped files detail:")
        for path, reason in skipped_files:
            print(f" - {path} ({reason})")

    print("=" * 60)
    return {
        "files_count": len(file_stats),
        "total_tokens": combined_tokens,
        "total_chars": total_chars,
        "total_words": total_words,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate files in a directory and count tokens with tiktoken."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="docs/",
        help="Target folder path to scan (default: docs/)",
    )
    parser.add_argument(
        "--encoding",
        "-e",
        default="cl100k_base",
        help="tiktoken encoding to use (default: cl100k_base, e.g. o200k_base, p50k_base)",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Do not display per-file breakdown table",
    )
    args = parser.parse_args()

    scan_and_count(
        folder_path=args.folder,
        encoding_name=args.encoding,
        show_details=not args.no_details,
    )


if __name__ == "__main__":
    main()
