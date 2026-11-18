"""Commit Diff & AST Analyzer.
Connects Git comparison with AST parsing to identify added and modified classes,
functions, methods, and docstrings since a specific git commit.
Allows Agentic AI systems to query and update documentation without surfing large codebases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Enable running both as module and standalone script
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from .ast_graph import ClassInfo, CodeGraph, FunctionInfo, ModuleInfo
    from .git_compare import GitComparator, GitFileChange
except ImportError:
    from ast_graph import CodeGraph
    from git_compare import GitComparator


@dataclass
class ChangedSymbol:
    symbol_type: str  # 'module', 'class', 'method', 'function'
    name: str
    qualname: str
    file_path: str
    change_status: str  # 'ADDED', 'MODIFIED'
    signature: str | None = None
    docstring: str | None = None
    bases: list[str] = field(default_factory=list)
    lineno: int = 0
    end_lineno: int = 0
    calls: list[str] = field(default_factory=list)
    parent_class: str | None = None


@dataclass
class CommitAnalysisResult:
    base_commit: str
    target: str
    changed_modules: list[str]
    added_classes: list[ChangedSymbol]
    modified_classes: list[ChangedSymbol]
    added_functions: list[ChangedSymbol]
    modified_functions: list[ChangedSymbol]
    missing_docstrings: list[ChangedSymbol]
    module_summaries: dict[str, dict[str, Any]]


class CommitDiffAnalyzer:
    def __init__(self, repo_root: str | None = None):
        self.repo_root = Path(repo_root or os.getcwd()).resolve()
        self.git_comparator = GitComparator(repo_root=self.repo_root)
        self.code_graph = CodeGraph(repo_root=self.repo_root)

    def analyze(
        self,
        base_commit: str,
        target: str = "HEAD",
        path_prefix: str | None = None,
        extension_filter: list[str] | None = None,
    ) -> CommitAnalysisResult:
        exts = extension_filter or [".py"]
        git_changes = self.git_comparator.compare(
            base_commit=base_commit,
            target=target,
            path_prefix=path_prefix,
            extension_filter=exts,
            ignore_deleted=True,
        )

        added_classes: list[ChangedSymbol] = []
        modified_classes: list[ChangedSymbol] = []
        added_functions: list[ChangedSymbol] = []
        modified_functions: list[ChangedSymbol] = []
        missing_docstrings: list[ChangedSymbol] = []
        module_summaries: dict[str, dict[str, Any]] = {}
        changed_modules: list[str] = []

        for change in git_changes:
            rel_path = change.file_path
            if not rel_path.endswith(".py"):
                continue

            mod_info = self.code_graph.parse_file(rel_path)
            if not mod_info:
                continue

            changed_modules.append(rel_path)
            is_new_file = change.status == "A"

            mod_summary = {
                "file_path": rel_path,
                "module_name": mod_info.module_name,
                "status": "ADDED" if is_new_file else "MODIFIED",
                "docstring": mod_info.docstring,
                "classes": {},
                "functions": {},
            }

            # Check classes & methods
            for cls_name, cls in mod_info.classes.items():
                cls_is_new = is_new_file or change.line_in_changed_ranges(
                    cls.lineno, cls.end_lineno
                )
                if not cls_is_new:
                    # Check if any method in this class was modified
                    any_method_changed = any(
                        change.line_in_changed_ranges(m.lineno, m.end_lineno)
                        for m in cls.methods.values()
                    )
                    if not any_method_changed:
                        continue

                cls_symbol = ChangedSymbol(
                    symbol_type="class",
                    name=cls.name,
                    qualname=cls.qualname,
                    file_path=rel_path,
                    change_status="ADDED" if is_new_file else "MODIFIED",
                    docstring=cls.docstring,
                    bases=cls.bases,
                    lineno=cls.lineno,
                    end_lineno=cls.end_lineno,
                )

                if is_new_file:
                    added_classes.append(cls_symbol)
                else:
                    modified_classes.append(cls_symbol)

                if not cls.docstring:
                    missing_docstrings.append(cls_symbol)

                # Analyze methods
                meth_summaries = {}
                for meth_name, meth in cls.methods.items():
                    meth_changed = is_new_file or change.line_in_changed_ranges(
                        meth.lineno, meth.end_lineno
                    )
                    if meth_changed:
                        meth_symbol = ChangedSymbol(
                            symbol_type="method",
                            name=meth.name,
                            qualname=meth.qualname,
                            file_path=rel_path,
                            change_status="ADDED" if is_new_file else "MODIFIED",
                            signature=meth.signature,
                            docstring=meth.docstring,
                            lineno=meth.lineno,
                            end_lineno=meth.end_lineno,
                            calls=meth.calls,
                            parent_class=cls.name,
                        )
                        if is_new_file:
                            added_functions.append(meth_symbol)
                        else:
                            modified_functions.append(meth_symbol)

                        if not meth.docstring and not meth.name.startswith("__"):
                            missing_docstrings.append(meth_symbol)

                        meth_summaries[meth.name] = {
                            "signature": meth.signature,
                            "docstring": meth.docstring,
                            "status": "ADDED" if is_new_file else "MODIFIED",
                        }

                mod_summary["classes"][cls.name] = {
                    "bases": cls.bases,
                    "docstring": cls.docstring,
                    "methods": meth_summaries,
                }

            # Check top-level functions
            for func_name, func in mod_info.functions.items():
                func_changed = is_new_file or change.line_in_changed_ranges(
                    func.lineno, func.end_lineno
                )
                if func_changed:
                    func_symbol = ChangedSymbol(
                        symbol_type="function",
                        name=func.name,
                        qualname=func.qualname,
                        file_path=rel_path,
                        change_status="ADDED" if is_new_file else "MODIFIED",
                        signature=func.signature,
                        docstring=func.docstring,
                        lineno=func.lineno,
                        end_lineno=func.end_lineno,
                        calls=func.calls,
                    )
                    if is_new_file:
                        added_functions.append(func_symbol)
                    else:
                        modified_functions.append(func_symbol)

                    if not func.docstring:
                        missing_docstrings.append(func_symbol)

                    mod_summary["functions"][func.name] = {
                        "signature": func.signature,
                        "docstring": func.docstring,
                        "status": "ADDED" if is_new_file else "MODIFIED",
                    }

            module_summaries[rel_path] = mod_summary

        return CommitAnalysisResult(
            base_commit=base_commit,
            target=target,
            changed_modules=changed_modules,
            added_classes=added_classes,
            modified_classes=modified_classes,
            added_functions=added_functions,
            modified_functions=modified_functions,
            missing_docstrings=missing_docstrings,
            module_summaries=module_summaries,
        )

    def generate_markdown_manifest(self, result: CommitAnalysisResult) -> str:
        """Generates comprehensive markdown documentation manifest for the changes."""
        lines: list[str] = [
            f"# API Documentation Manifest (Diff from {result.base_commit[:8]} to {result.target})",
            "",
            "## Summary",
            f"- **Modules Changed**: {len(result.changed_modules)}",
            f"- **Classes Added**: {len(result.added_classes)}",
            f"- **Classes Modified**: {len(result.modified_classes)}",
            f"- **Functions/Methods Added**: {len(result.added_functions)}",
            f"- **Functions/Methods Modified**: {len(result.modified_functions)}",
            f"- **Undocumented Symbols**: {len(result.missing_docstrings)}",
            "",
            "---",
            "",
            "## Module Details & API Reference",
            "",
        ]

        for file_path, mod in sorted(result.module_summaries.items()):
            lines.append(f"### `{mod['module_name']}` ({file_path})")
            lines.append(f"**Status**: `{mod['status']}`")
            if mod["docstring"]:
                lines.append(f"> {mod['docstring'].strip()}")
            lines.append("")

            # Classes
            if mod["classes"]:
                lines.append("#### Classes")
                for cls_name, cls_info in mod["classes"].items():
                    bases_str = (
                        f"({', '.join(cls_info['bases'])})" if cls_info["bases"] else ""
                    )
                    lines.append(f"- **`class {cls_name}{bases_str}`**")
                    if cls_info["docstring"]:
                        lines.append(
                            f"  - *Doc*: {cls_info['docstring'].strip().replace(chr(10), ' ')}"
                        )
                    for m_name, m_info in cls_info["methods"].items():
                        lines.append(
                            f"  - `{m_info['status']}` `def {m_name}{m_info['signature']}`"
                        )
                        if m_info["docstring"]:
                            first_doc = m_info["docstring"].strip().splitlines()[0]
                            lines.append(f"    - *Doc*: {first_doc}")
                lines.append("")

            # Top-level functions
            if mod["functions"]:
                lines.append("#### Functions")
                for f_name, f_info in mod["functions"].items():
                    lines.append(
                        f"- `{f_info['status']}` `def {f_name}{f_info['signature']}`"
                    )
                    if f_info["docstring"]:
                        lines.append(
                            f"  - *Doc*: {f_info['docstring'].strip().replace(chr(10), ' ')}"
                        )
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AST & Git Commit API Diff Inspector")
    parser.add_argument(
        "--commit", required=True, help="Base commit hash to compare against"
    )
    parser.add_argument(
        "--target", default="HEAD", help="Target commit/branch (default: HEAD)"
    )
    parser.add_argument(
        "--path", help="Filter by path prefix (e.g. src/neuron_toolkit)"
    )
    parser.add_argument("--summary", action="store_true", help="Print summary counts")
    parser.add_argument(
        "--manifest", action="store_true", help="Generate full markdown docs manifest"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON analysis")
    parser.add_argument("--inspect-symbol", help="Inspect a specific symbol name")
    parser.add_argument(
        "--inspect-module", help="Inspect a specific module name or path"
    )
    parser.add_argument(
        "--missing-docs",
        action="store_true",
        help="List all undocumented new/modified symbols",
    )

    args = parser.parse_args()

    analyzer = CommitDiffAnalyzer()
    res = analyzer.analyze(
        base_commit=args.commit,
        target=args.target,
        path_prefix=args.path,
    )

    if args.summary:
        print(f"=== API Diff Analysis ({args.commit[:10]} -> {args.target}) ===")
        print(f"Changed Modules:           {len(res.changed_modules)}")
        print(f"Added Classes:             {len(res.added_classes)}")
        print(f"Modified Classes:          {len(res.modified_classes)}")
        print(f"Added Functions/Methods:   {len(res.added_functions)}")
        print(f"Modified Functions/Methods:{len(res.modified_functions)}")
        print(f"Missing Docstrings:        {len(res.missing_docstrings)}")
        return

    if args.inspect_symbol:
        sym_name = args.inspect_symbol.lower()
        matches = [
            s
            for s in res.added_classes
            + res.modified_classes
            + res.added_functions
            + res.modified_functions
            if sym_name in s.name.lower() or sym_name in s.qualname.lower()
        ]
        if not matches:
            print(f"No changed symbol found matching '{args.inspect_symbol}'")
            return
        print(
            f"Found {len(matches)} matching symbol(s) changed since {args.commit[:10]}:"
        )
        for s in matches:
            print(f"[{s.symbol_type.upper():8}] {s.qualname}")
            print(f"  Status:    {s.change_status}")
            print(f"  File:      {s.file_path}:{s.lineno}-{s.end_lineno}")
            if s.signature:
                print(f"  Signature: def {s.name}{s.signature}")
            if s.bases:
                print(f"  Bases:     {', '.join(s.bases)}")
            print(f"  Docstring: {s.docstring or 'None'}")
            if s.calls:
                print(f"  Calls:     {', '.join(s.calls[:8])}")
            print("-" * 60)
        return

    if args.inspect_module:
        mod_query = args.inspect_module
        matched_mods = {
            k: v
            for k, v in res.module_summaries.items()
            if mod_query in k or mod_query in v["module_name"]
        }
        if not matched_mods:
            print(f"No module found matching '{mod_query}'")
            return
        for path, summary in matched_mods.items():
            print(f"Module: {summary['module_name']} ({path})")
            print(f"Status: {summary['status']}")
            print(f"Docstring: {summary['docstring'] or 'None'}")
            print(f"Classes ({len(summary['classes'])}):")
            for cname, cdata in summary["classes"].items():
                print(f"  class {cname}({', '.join(cdata['bases'])}):")
                if cdata["docstring"]:
                    print(f"    Doc: {cdata['docstring'].strip().splitlines()[0]}")
                for mname, mdata in cdata["methods"].items():
                    print(f"    - def {mname}{mdata['signature']}")
            print(f"Functions ({len(summary['functions'])}):")
            for fname, fdata in summary["functions"].items():
                print(f"  def {fname}{fdata['signature']}")
                if fdata["docstring"]:
                    print(f"    Doc: {fdata['docstring'].strip().splitlines()[0]}")
            print("=" * 60)
        return

    if args.missing_docs:
        print(f"=== Undocumented Symbols Changed Since {args.commit[:10]} ===")
        for s in res.missing_docstrings:
            sig = f"{s.signature}" if s.signature else ""
            print(f"[{s.symbol_type:8}] {s.qualname}{sig} ({s.file_path}:{s.lineno})")
        return

    if args.manifest:
        manifest = analyzer.generate_markdown_manifest(res)
        print(manifest)
        return

    if args.json:
        out = {
            "base_commit": res.base_commit,
            "target": res.target,
            "changed_modules": res.changed_modules,
            "added_classes": [asdict(s) for s in res.added_classes],
            "modified_classes": [asdict(s) for s in res.modified_classes],
            "added_functions": [asdict(s) for s in res.added_functions],
            "modified_functions": [asdict(s) for s in res.modified_functions],
            "missing_docstrings": [asdict(s) for s in res.missing_docstrings],
            "module_summaries": res.module_summaries,
        }
        print(json.dumps(out, indent=2))
        return

    # Default: print brief overview and help
    print(f"=== Commit Diff Analyzer: {args.commit[:10]} -> {args.target} ===")
    print(f"Total Changed Modules: {len(res.changed_modules)}")
    print(
        f"Added Classes: {len(res.added_classes)}, Modified Classes: {len(res.modified_classes)}"
    )
    print(
        f"Added Functions: {len(res.added_functions)}, Modified Functions: {len(res.modified_functions)}"
    )
    print("\nOptions:")
    print("  --manifest            Generate full markdown documentation manifest")
    print("  --summary             View summary table")
    print("  --inspect-symbol <sym> Inspect full signature & docstring of a symbol")
    print("  --inspect-module <mod> Inspect all classes/methods in a module")
    print("  --missing-docs        List items lacking docstrings")
    print("  --json                Full JSON dump")


if __name__ == "__main__":
    main()
