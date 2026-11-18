"""Git comparison utility for tracking changes between commits and working tree.
Directly compares a base commit to target (HEAD or working tree) ignoring intermediate deleted files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)

    @property
    def new_line_range(self) -> tuple[int, int]:
        """Returns (start_line, end_line) in the target/new file."""
        end = self.new_start + max(self.new_count, 1) - 1
        return (self.new_start, end)


@dataclass
class GitFileChange:
    file_path: str
    status: str  # 'A', 'M', 'D', 'R', etc.
    insertions: int
    deletions: int
    exists_on_disk: bool
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def changed_line_ranges(self) -> list[tuple[int, int]]:
        """List of (start, end) line ranges modified or added in the new file."""
        return [hunk.new_line_range for hunk in self.hunks]

    def line_in_changed_ranges(
        self, lineno: int, end_lineno: int | None = None
    ) -> bool:
        """Check if a line or line range overlaps with any diff hunk."""
        if self.status == "A":
            return True
        if not self.hunks:
            return True
        end = end_lineno if end_lineno is not None else lineno
        for start_range, end_range in self.changed_line_ranges:
            if not (end < start_range or lineno > end_range):
                return True
        return False


class GitComparator:
    def __init__(self, repo_root: str | None = None):
        self.repo_root = Path(repo_root or os.getcwd()).resolve()

    def _run_git(self, args: list[str]) -> str:
        cmd = ["git"] + args
        res = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"Git command failed: {' '.join(cmd)}\nError: {res.stderr.strip()}"
            )
        return res.stdout

    def parse_hunks(self, diff_text: str) -> list[Hunk]:
        hunks: list[Hunk] = []
        current_hunk: Hunk | None = None
        hunk_header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

        for line in diff_text.splitlines():
            if line.startswith("@@"):
                m = hunk_header_re.match(line)
                if m:
                    old_start = int(m.group(1))
                    old_count = int(m.group(2)) if m.group(2) is not None else 1
                    new_start = int(m.group(3))
                    new_count = int(m.group(4)) if m.group(4) is not None else 1
                    current_hunk = Hunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                    )
                    hunks.append(current_hunk)
            elif current_hunk is not None:
                current_hunk.lines.append(line)

        return hunks

    def compare(
        self,
        base_commit: str,
        target: str | None = None,
        path_prefix: str | None = None,
        extension_filter: list[str] | None = None,
        ignore_deleted: bool = False,
    ) -> list[GitFileChange]:
        """Directly compare base_commit against target (or current working tree if target is None).
        Direct comparison ensures intermediate files added and deleted in intermediate commits
        are NOT counted.
        """
        diff_target = [f"{base_commit}..{target}"] if target else [base_commit]

        # 1. Get name-status
        name_status_cmd = ["diff", "--name-status"] + diff_target
        if path_prefix:
            name_status_cmd.extend(["--", path_prefix])
        name_status_out = self._run_git(name_status_cmd)

        status_map: dict[str, str] = {}
        for line in name_status_out.strip().splitlines():
            if not line:
                continue
            parts = line.split(maxsplit=2)
            st = parts[0]
            # Handle rename e.g. R100 old_path new_path
            if st.startswith("R"):
                file_path = parts[2] if len(parts) > 2 else parts[1]
                status_map[file_path] = "R"
            else:
                file_path = parts[1]
                status_map[file_path] = st[0]

        # 2. Get numstat (lines added/deleted)
        numstat_cmd = ["diff", "--numstat"] + diff_target
        if path_prefix:
            numstat_cmd.extend(["--", path_prefix])
        numstat_out = self._run_git(numstat_cmd)

        numstat_map: dict[str, tuple[int, int]] = {}
        for line in numstat_out.strip().splitlines():
            if not line:
                continue
            parts = line.split(maxsplit=2)
            if len(parts) >= 3:
                ins = int(parts[0]) if parts[0] != "-" else 0
                dele = int(parts[1]) if parts[1] != "-" else 0
                file_path = parts[2]
                # In case of rename like 'foo => bar'
                if " => " in file_path:
                    # Clean up git rename format
                    file_path = re.sub(r"\{.*? => (.*?)\}", r"\1", file_path)
                    file_path = file_path.split(" => ")[-1]
                numstat_map[file_path] = (ins, dele)

        results: list[GitFileChange] = []

        all_files = sorted(set(list(status_map.keys()) + list(numstat_map.keys())))
        for file_path in all_files:
            # Extension filter
            if extension_filter:
                if not any(file_path.endswith(ext) for ext in extension_filter):
                    continue

            status = status_map.get(file_path, "M")
            ins, dele = numstat_map.get(file_path, (0, 0))
            full_path = self.repo_root / file_path
            exists = full_path.exists()

            if ignore_deleted and (status == "D" or not exists):
                continue

            # Parse hunks for changed line ranges
            hunks: list[Hunk] = []
            if status != "D" and exists:
                hunk_diff_cmd = ["diff", "-U0"] + diff_target + ["--", file_path]
                diff_text = self._run_git(hunk_diff_cmd)
                hunks = self.parse_hunks(diff_text)

            results.append(
                GitFileChange(
                    file_path=file_path,
                    status=status,
                    insertions=ins,
                    deletions=dele,
                    exists_on_disk=exists,
                    hunks=hunks,
                )
            )

        return results

    def get_summary(self, changes: list[GitFileChange]) -> dict[str, int]:
        summary = {
            "added": 0,
            "modified": 0,
            "deleted": 0,
            "renamed": 0,
            "total_insertions": 0,
            "total_deletions": 0,
        }
        for c in changes:
            if c.status == "A":
                summary["added"] += 1
            elif c.status == "M":
                summary["modified"] += 1
            elif c.status == "D":
                summary["deleted"] += 1
            elif c.status == "R":
                summary["renamed"] += 1
            summary["total_insertions"] += c.insertions
            summary["total_deletions"] += c.deletions
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Git Compare Utility for Agentic Workflows"
    )
    parser.add_argument(
        "--commit", required=True, help="Base commit hash to compare from"
    )
    parser.add_argument(
        "--target", default="HEAD", help="Target commit/ref (default: HEAD)"
    )
    parser.add_argument("--path", help="Filter changes by directory/file path")
    parser.add_argument(
        "--ext", nargs="*", help="Filter changes by file extension (e.g. .py)"
    )
    parser.add_argument(
        "--ignore-deleted", action="store_true", help="Omit deleted/nonexistent files"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--summary", action="store_true", help="Print summary only")

    args = parser.parse_args()

    comparator = GitComparator()
    changes = comparator.compare(
        base_commit=args.commit,
        target=args.target,
        path_prefix=args.path,
        extension_filter=args.ext,
        ignore_deleted=args.ignore_deleted,
    )

    if args.json:
        data = {
            "base_commit": args.commit,
            "target": args.target,
            "summary": comparator.get_summary(changes),
            "changes": [
                {
                    "file_path": c.file_path,
                    "status": c.status,
                    "insertions": c.insertions,
                    "deletions": c.deletions,
                    "exists_on_disk": c.exists_on_disk,
                    "changed_line_ranges": c.changed_line_ranges,
                }
                for c in changes
            ],
        }
        print(json.dumps(data, indent=2))
        return

    summary = comparator.get_summary(changes)
    print(f"=== Git Compare: {args.commit} -> {args.target} ===")
    print(
        f"Files: {len(changes)} | Added: {summary['added']} | Modified: {summary['modified']} | "
        f"Deleted: {summary['deleted']} | Lines: +{summary['total_insertions']}/-{summary['total_deletions']}"
    )
    print("-" * 70)

    if not args.summary:
        for c in changes:
            status_desc = {
                "A": "ADDED",
                "M": "MODIFIED",
                "D": "DELETED",
                "R": "RENAMED",
            }.get(c.status, c.status)
            ranges_str = (
                f" [lines: {', '.join(f'{s}-{e}' for s, e in c.changed_line_ranges[:5])}{'...' if len(c.changed_line_ranges) > 5 else ''}]"
                if c.changed_line_ranges
                else ""
            )
            print(
                f"[{status_desc:8}] {c.file_path:<50} (+{c.insertions}/-{c.deletions}){ranges_str}"
            )


if __name__ == "__main__":
    main()
