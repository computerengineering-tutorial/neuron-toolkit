"""AST-based Codebase Graph and Codemap Generator.
Extracts modules, classes, functions, signatures, docstrings, calls, and line ranges.
Supports querying symbols, finding undocumented nodes, and building call graphs.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CallInfo:
    caller: str
    callee: str
    lineno: int


@dataclass
class FunctionInfo:
    name: str
    qualname: str
    signature: str
    docstring: str | None
    lineno: int
    end_lineno: int
    is_async: bool = False
    is_method: bool = False
    parent_class: str | None = None
    decorators: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    returns: str | None = None
    args: list[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    qualname: str
    docstring: str | None
    bases: list[str]
    lineno: int
    end_lineno: int
    decorators: list[str] = field(default_factory=list)
    methods: dict[str, FunctionInfo] = field(default_factory=dict)


@dataclass
class ModuleInfo:
    file_path: str
    module_name: str
    docstring: str | None
    imports: list[str] = field(default_factory=list)
    classes: dict[str, ClassInfo] = field(default_factory=dict)
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    total_lines: int = 0


class CallVisitor(ast.NodeVisitor):
    """Collects all function and method call names inside an AST node."""

    def __init__(self):
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call):
        func_name = self._resolve_name(node.func)
        if func_name:
            self.calls.append(func_name)
        self.generic_visit(node)

    def _resolve_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            val = self._resolve_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        return None


class CodebaseASTVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, module_name: str):
        self.file_path = file_path
        self.module_name = module_name
        self.module_info = ModuleInfo(
            file_path=file_path,
            module_name=module_name,
            docstring=None,
        )
        self.current_class: ClassInfo | None = None

    def _format_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> tuple[str, list[str], str | None]:
        # Parameter list
        arg_strs: list[str] = []

        # Positional-only args
        for arg in node.args.posonlyargs:
            s = arg.arg
            if arg.annotation:
                s += f": {ast.unparse(arg.annotation)}"
            arg_strs.append(s)
        if node.args.posonlyargs:
            arg_strs.append("/")

        # Standard positional args and defaults
        defaults_start = len(node.args.args) - len(node.args.defaults)
        for i, arg in enumerate(node.args.args):
            s = arg.arg
            if arg.annotation:
                s += f": {ast.unparse(arg.annotation)}"
            if i >= defaults_start:
                default_val = node.args.defaults[i - defaults_start]
                s += f" = {ast.unparse(default_val)}"
            arg_strs.append(s)

        # Vararg (*args)
        if node.args.vararg:
            s = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                s += f": {ast.unparse(node.args.vararg.annotation)}"
            arg_strs.append(s)
        elif node.args.kwonlyargs:
            arg_strs.append("*")

        # Kwonly args and kw_defaults
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            s = arg.arg
            if arg.annotation:
                s += f": {ast.unparse(arg.annotation)}"
            if default is not None:
                s += f" = {ast.unparse(default)}"
            arg_strs.append(s)

        # Kwarg (**kwargs)
        if node.args.kwarg:
            s = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                s += f": {ast.unparse(node.args.kwarg.annotation)}"
            arg_strs.append(s)

        # Return annotation
        ret_str = ast.unparse(node.returns) if node.returns else None

        sig = f"({', '.join(arg_strs)})"
        if ret_str:
            sig += f" -> {ret_str}"

        return sig, arg_strs, ret_str

    def visit_Module(self, node: ast.Module):
        self.module_info.docstring = ast.get_docstring(node)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.name if not alias.asname else f"{alias.name} as {alias.asname}"
            self.module_info.imports.append(f"import {name}")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        aliases = [
            a.name if not a.asname else f"{a.name} as {a.asname}" for a in node.names
        ]
        self.module_info.imports.append(f"from {mod} import {', '.join(aliases)}")

    def visit_ClassDef(self, node: ast.ClassDef):
        bases = [ast.unparse(b) for b in node.bases]
        decorators = [ast.unparse(d) for d in node.decorator_list]
        docstring = ast.get_docstring(node)

        qualname = f"{self.module_name}.{node.name}" if self.module_name else node.name
        class_info = ClassInfo(
            name=node.name,
            qualname=qualname,
            docstring=docstring,
            bases=bases,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            decorators=decorators,
        )

        prev_class = self.current_class
        self.current_class = class_info

        # Visit children (methods, etc.)
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(item)

        self.current_class = prev_class
        self.module_info.classes[node.name] = class_info

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        sig, args, ret = self._format_signature(node)
        docstring = ast.get_docstring(node)
        decorators = [ast.unparse(d) for d in node.decorator_list]

        call_visitor = CallVisitor()
        for item in node.body:
            call_visitor.visit(item)

        is_method = self.current_class is not None
        parent_class_name = self.current_class.name if self.current_class else None

        if is_method:
            qualname = f"{self.current_class.qualname}.{node.name}"
        else:
            qualname = (
                f"{self.module_name}.{node.name}" if self.module_name else node.name
            )

        func_info = FunctionInfo(
            name=node.name,
            qualname=qualname,
            signature=sig,
            docstring=docstring,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            parent_class=parent_class_name,
            decorators=decorators,
            calls=sorted(list(set(call_visitor.calls))),
            returns=ret,
            args=args,
        )

        if is_method and self.current_class:
            self.current_class.methods[node.name] = func_info
        else:
            self.module_info.functions[node.name] = func_info

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)


class CodeGraph:
    def __init__(self, repo_root: str | None = None):
        self.repo_root = Path(repo_root or os.getcwd()).resolve()
        self.modules: dict[str, ModuleInfo] = {}

    def _relpath_to_modname(self, rel_path: str) -> str:
        p = Path(rel_path)
        parts = list(p.parts)
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts and parts[-1] == "__init__":
            parts.pop()
        # Strip common src/ prefix if present
        if parts and parts[0] == "src":
            parts.pop(0)
        return ".".join(parts)

    def parse_file(self, rel_path: str) -> ModuleInfo | None:
        full_path = self.repo_root / rel_path
        if not full_path.exists() or not full_path.is_file():
            return None

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(content, filename=str(full_path))
        except Exception as e:
            print(f"Error parsing {rel_path}: {e}")
            return None

        mod_name = self._relpath_to_modname(rel_path)
        visitor = CodebaseASTVisitor(rel_path, mod_name)
        visitor.visit(tree)
        visitor.module_info.total_lines = len(content.splitlines())

        self.modules[rel_path] = visitor.module_info
        return visitor.module_info

    def parse_directory(
        self,
        directory: str = "src",
        exclude_patterns: list[str] | None = None,
    ):
        target_dir = self.repo_root / directory
        if not target_dir.exists():
            return

        excludes = exclude_patterns or [
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "tests",
        ]
        for p in target_dir.rglob("*.py"):
            rel = str(p.relative_to(self.repo_root))
            if any(exc in rel for exc in excludes):
                continue
            self.parse_file(rel)

    def query_symbols(
        self,
        name_pattern: str | None = None,
        doc_keyword: str | None = None,
        symbol_type: str | None = None,  # 'class', 'function', 'method'
        missing_doc: bool = False,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        name_re = re.compile(name_pattern, re.IGNORECASE) if name_pattern else None
        doc_re = re.compile(doc_keyword, re.IGNORECASE) if doc_keyword else None

        for file_path, mod in self.modules.items():
            # Check classes
            if not symbol_type or symbol_type == "class":
                for cls_name, cls in mod.classes.items():
                    if name_re and not name_re.search(cls.name):
                        continue
                    if doc_re and (
                        not cls.docstring or not doc_re.search(cls.docstring)
                    ):
                        continue
                    if missing_doc and cls.docstring:
                        continue
                    results.append(
                        {
                            "type": "class",
                            "name": cls.name,
                            "qualname": cls.qualname,
                            "file": file_path,
                            "lines": (cls.lineno, cls.end_lineno),
                            "docstring": cls.docstring,
                            "bases": cls.bases,
                        }
                    )

            # Check methods
            if not symbol_type or symbol_type == "method":
                for cls in mod.classes.values():
                    for meth_name, meth in cls.methods.items():
                        if name_re and not name_re.search(meth.name):
                            continue
                        if doc_re and (
                            not meth.docstring or not doc_re.search(meth.docstring)
                        ):
                            continue
                        if missing_doc and meth.docstring:
                            continue
                        results.append(
                            {
                                "type": "method",
                                "name": meth.name,
                                "qualname": meth.qualname,
                                "file": file_path,
                                "lines": (meth.lineno, meth.end_lineno),
                                "signature": meth.signature,
                                "docstring": meth.docstring,
                                "calls": meth.calls,
                            }
                        )

            # Check top-level functions
            if not symbol_type or symbol_type == "function":
                for func_name, func in mod.functions.items():
                    if name_re and not name_re.search(func.name):
                        continue
                    if doc_re and (
                        not func.docstring or not doc_re.search(func.docstring)
                    ):
                        continue
                    if missing_doc and func.docstring:
                        continue
                    results.append(
                        {
                            "type": "function",
                            "name": func.name,
                            "qualname": func.qualname,
                            "file": file_path,
                            "lines": (func.lineno, func.end_lineno),
                            "signature": func.signature,
                            "docstring": func.docstring,
                            "calls": func.calls,
                        }
                    )

        return results

    def generate_codemap(self, filter_file: str | None = None) -> str:
        """Generate a hierarchical ASCII codemap of modules, classes, and methods."""
        lines: list[str] = []

        modules_to_show = (
            {k: v for k, v in self.modules.items() if filter_file in k}
            if filter_file
            else self.modules
        )

        for rel_path in sorted(modules_to_show.keys()):
            mod = modules_to_show[rel_path]
            lines.append(f"📁 {rel_path} ({mod.module_name})")
            if mod.docstring:
                first_line = mod.docstring.strip().splitlines()[0]
                lines.append(f'   │  """{first_line}"""')

            # Standalone functions
            for func in mod.functions.values():
                dec = f" [{' '.join(func.decorators)}]" if func.decorators else ""
                lines.append(f"   ├── ⚡ def {func.name}{func.signature}{dec}")
                if func.docstring:
                    first = func.docstring.strip().splitlines()[0]
                    lines.append(f"   │      Doc: {first}")

            # Classes
            for cls in mod.classes.values():
                bases = f"({', '.join(cls.bases)})" if cls.bases else ""
                lines.append(f"   ├── 🏛️ class {cls.name}{bases}:")
                if cls.docstring:
                    first = cls.docstring.strip().splitlines()[0]
                    lines.append(f"   │   │  Doc: {first}")
                for meth in cls.methods.values():
                    dec = f" [{' '.join(meth.decorators)}]" if meth.decorators else ""
                    lines.append(f"   │   ├── def {meth.name}{meth.signature}{dec}")
                    if meth.docstring:
                        first = meth.docstring.strip().splitlines()[0]
                        lines.append(f"   │   │      Doc: {first}")

            lines.append("   │")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AST-based Code Graph & Codemap")
    parser.add_argument(
        "--dir", default="src", help="Directory to parse (default: src)"
    )
    parser.add_argument("--file", help="Specific file to parse")
    parser.add_argument(
        "--codemap", action="store_true", help="Print codemap hierarchy"
    )
    parser.add_argument("--query", help="Query symbol name (regex)")
    parser.add_argument("--doc-query", help="Search docstring keyword")
    parser.add_argument(
        "--missing-docs",
        action="store_true",
        help="Find functions/classes missing docstrings",
    )
    parser.add_argument("--json", action="store_true", help="Output in JSON")

    args = parser.parse_args()

    graph = CodeGraph()
    if args.file:
        graph.parse_file(args.file)
    else:
        graph.parse_directory(args.dir)

    if args.query or args.doc_query or args.missing_docs:
        results = graph.query_symbols(
            name_pattern=args.query,
            doc_keyword=args.doc_query,
            missing_doc=args.missing_docs,
        )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"Found {len(results)} matches:")
            for r in results:
                doc = (
                    r["docstring"].strip().splitlines()[0]
                    if r["docstring"]
                    else "NO DOCSTRING"
                )
                sig = f" {r.get('signature', '')}" if "signature" in r else ""
                print(f"[{r['type'].upper():8}] {r['qualname']}{sig}")
                print(f"           File: {r['file']}:{r['lines'][0]}-{r['lines'][1]}")
                print(f"           Doc:  {doc}")
                if r.get("calls"):
                    print(f"           Calls: {', '.join(r['calls'][:8])}")
                print()
        return

    if args.codemap:
        print(graph.generate_codemap(filter_file=args.file))
        return

    print(f"Parsed {len(graph.modules)} Python files in {args.dir}.")
    print("Use --codemap, --query <regex>, --doc-query <kw>, or --missing-docs.")


if __name__ == "__main__":
    main()
