"""Formal regression benchmarks using the pytest-benchmark plugin."""

from __future__ import annotations

import pytest

from neuron_toolkit import NeuronGraph
from neuron_toolkit.passes import (
    CommonSubexpressionElimination,
    ConstantFolding,
    DeadCodeElimination,
    PassManager,
)
from neuron_toolkit.pattern import Pattern

from .perf_utils import create_large_onnx_model


@pytest.fixture(scope="module")
def benchmark_graph() -> NeuronGraph:
    """Create a 500-block (~2,000 node) model once for benchmarking."""
    model_proto = create_large_onnx_model(num_blocks=500)
    return NeuronGraph(model_proto)


def test_benchmark_query_op(benchmark, benchmark_graph):
    """Benchmark NeuronQuery op filtering throughput."""
    def run_query():
        return benchmark_graph.query().op("Conv").all()

    result = benchmark(run_query)
    assert len(result) > 0


def test_benchmark_query_descendants(benchmark, benchmark_graph):
    """Benchmark ancestor/descendant graph traversal throughput."""
    root_conv = benchmark_graph.query().op("Conv").first()

    def run_traversal():
        return benchmark_graph.query().name(root_conv.name, exact=True).descendants().all()

    result = benchmark(run_traversal)
    assert len(result) > 0


def test_benchmark_pattern_match(benchmark, benchmark_graph):
    """Benchmark declarative Pattern matching on a 2,000 node graph."""
    pattern = Pattern.op("BatchNormalization", Pattern.op("Conv"))

    def run_match():
        return benchmark_graph.findall(pattern)

    matches = benchmark(run_match)
    assert len(matches) > 0


def test_benchmark_pass_manager_pipeline(benchmark, benchmark_graph):
    """Benchmark multi-pass PassManager optimization pipeline execution."""
    pm = PassManager([
        CommonSubexpressionElimination(),
        ConstantFolding(),
        DeadCodeElimination(),
    ])

    def run_pipeline():
        return pm.run(benchmark_graph)

    opt_g, stats = benchmark(run_pipeline)
    assert opt_g is not None
    assert stats is not None


def test_benchmark_json_export(benchmark, benchmark_graph):
    """Benchmark graph dictionary/JSON export serialization."""
    def run_export():
        return benchmark_graph.to_dict()

    d = benchmark(run_export)
    assert isinstance(d, dict)
    assert "nodes" in d
