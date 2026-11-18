"""Unit tests for AST-based CLI Subcommand in neuron-toolkit."""

from __future__ import annotations

import argparse

from neuron_toolkit.cli import ast_command


def test_cli_ast_codemap(capsys):
    """Test 'neuron ast --codemap' prints hierarchical codemap."""
    args = argparse.Namespace(
        command="ast",
        dir="src/neuron_toolkit/rules",
        file=None,
        codemap=True,
        query=None,
        doc_query=None,
        missing_docs=False,
        json=False,
    )
    ret = ast_command(args)
    assert ret == 0

    captured = capsys.readouterr()
    assert "📁" in captured.out
    assert "class" in captured.out or "def" in captured.out


def test_cli_ast_query(capsys):
    """Test 'neuron ast --query' searches symbols by regex."""
    args = argparse.Namespace(
        command="ast",
        dir="src/neuron_toolkit/compiler/passes/synthesis",
        file=None,
        codemap=False,
        query="gemma2",
        doc_query=None,
        missing_docs=False,
        json=False,
    )
    ret = ast_command(args)
    assert ret == 0

    captured = capsys.readouterr()
    assert "apply_gemma2_softcapping_fusion" in captured.out


def test_cli_ast_json(capsys):
    """Test 'neuron ast --query ... --json' outputs structured JSON."""
    args = argparse.Namespace(
        command="ast",
        dir="src/neuron_toolkit/compiler/passes/synthesis",
        file=None,
        codemap=False,
        query="softcap",
        doc_query=None,
        missing_docs=False,
        json=True,
    )
    ret = ast_command(args)
    assert ret == 0

    captured = capsys.readouterr()
    assert "[" in captured.out
    assert "apply_gemma2_softcapping_fusion" in captured.out
