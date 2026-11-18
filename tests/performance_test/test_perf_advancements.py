"""Performance benchmarks and verification tests for advanced compiler enhancements.

Validates:
1. Root-Op indexed pattern matching throughput on large models.
2. Precompiled execution plan forward-pass throughput in PyTorchRunner.
3. Opcode-indexed Equality Saturation convergence in EGraph.
4. Resilient external data handling for multi-gigabyte models.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuron_toolkit import NeuronGraph
from neuron_toolkit.egraph.core import EGraph, ENode
from neuron_toolkit.egraph.rules import saturate
from neuron_toolkit.pattern.detector import PatternDetector
from neuron_toolkit.pattern.dsl import Pattern
from neuron_toolkit.runtime.runner import PyTorchRunner
from .perf_utils import create_large_onnx_model


@pytest.fixture(scope="module")
def large_bench_model():
    """Create a 300-block (~1,200 node) model for execution benchmarks."""
    return create_large_onnx_model(num_blocks=300)


def test_benchmark_indexed_pattern_matching(benchmark, large_bench_model):
    """Benchmark declarative Pattern matching with root-op hash indexing."""
    graph = NeuronGraph(large_bench_model)
    det = PatternDetector(graph._shim())
    pattern = Pattern.op("BatchNormalization", Pattern.op("Conv"))

    def run_match():
        return det.find_all(pattern)

    matches = benchmark(run_match)
    assert len(matches) > 0


def test_benchmark_precompiled_pytorch_runner(benchmark):
    """Benchmark PyTorchRunner forward pass throughput with precompiled plan."""
    small_model = create_large_onnx_model(num_blocks=20)
    runner = PyTorchRunner(small_model)
    inp_name = runner.input_names[0]
    dummy_input = {inp_name: np.random.randn(1, 16, 32, 32).astype(np.float32)}

    def run_forward():
        return runner.run(dummy_input)

    result = benchmark(run_forward)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_benchmark_indexed_egraph_saturation(benchmark):
    """Benchmark Equality Saturation convergence using opcode-indexed rules."""
    def build_and_saturate():
        eg = EGraph()
        # Build expression: ((x + 0) * 1) + ((y * 1) + 0)
        c_x = eg.add(ENode(op="Var_x"))
        c_y = eg.add(ENode(op="Var_y"))
        c_0 = eg.add(ENode(op="Constant", attrs=(("value", 0.0),)))
        c_1 = eg.add(ENode(op="Constant", attrs=(("value", 1.0),)))

        x_plus_0 = eg.add(ENode(op="Add", children=(c_x, c_0)))
        x_mul_1 = eg.add(ENode(op="Mul", children=(x_plus_0, c_1)))

        y_mul_1 = eg.add(ENode(op="Mul", children=(c_y, c_1)))
        y_plus_0 = eg.add(ENode(op="Add", children=(y_mul_1, c_0)))

        root = eg.add(ENode(op="Add", children=(x_mul_1, y_plus_0)))
        eg.rebuild()
        stats = saturate(eg, max_iterations=6, node_limit=5000)
        return eg.find(root), stats

    root_id, stats = benchmark(build_and_saturate)
    assert root_id is not None
    assert stats["iterations"] >= 1


def test_lazy_tensor_map_resilience():
    """Verify LazyTensorMap gracefully handles unlocatable/corrupted initializers."""
    from unittest.mock import MagicMock
    import onnx
    from neuron_toolkit.backends.onnx.parser import LazyTensorMap

    bad_init = MagicMock(spec=onnx.TensorProto)
    bad_init.name = "corrupted_weight"

    tensor_map = LazyTensorMap([bad_init])
    arr = tensor_map["corrupted_weight"]
    assert isinstance(arr, np.ndarray)
    assert arr.size == 0
