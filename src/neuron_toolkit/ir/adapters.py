"""Backend adapters to convert between native formats and IRGraph."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from neuron_toolkit.ir.core import IRGraph, IRNode, IRTensor


@runtime_checkable
class BackendAdapter(Protocol):
    """Protocol for framework ingestion and emission."""

    @classmethod
    def can_handle(cls, source: Any) -> bool:
        """Check if this adapter can parse the provided source."""
        ...

    def import_to_ir(self, source: Any) -> IRGraph:
        """Convert external model representation into IRGraph."""
        ...

    def export_from_ir(self, graph: IRGraph, path: str | None = None) -> Any:
        """Convert IRGraph back into native model representation."""
        ...


class ONNXAdapter:
    """Adapter for ONNX models."""

    @classmethod
    def can_handle(cls, source: Any) -> bool:
        """Check if source is an ONNX ModelProto."""
        try:
            import onnx

            return isinstance(source, onnx.ModelProto)
        except ImportError:
            return False

    def import_to_ir(self, source: Any) -> IRGraph:
        """Convert onnx.ModelProto into IRGraph."""
        import onnx
        from onnx import numpy_helper

        if not isinstance(source, onnx.ModelProto):
            msg = f"Expected onnx.ModelProto, got {type(source)}"
            raise TypeError(msg)

        graph_proto = source.graph
        ir = IRGraph(name=graph_proto.name or "onnx_model")

        # Inputs
        for inp in graph_proto.input:
            ir.inputs.append(inp.name)

        # Outputs
        for out in graph_proto.output:
            ir.outputs.append(out.name)

        # Initializers
        for init in graph_proto.initializer:
            arr = numpy_helper.to_array(init)
            tensor = IRTensor(
                name=init.name,
                shape=tuple(arr.shape),
                dtype=str(arr.dtype),
                data=arr,
                is_initializer=True,
            )
            ir.add_tensor(tensor)

        # Value info
        for vi in list(graph_proto.value_info) + list(graph_proto.input) + list(graph_proto.output):
            if vi.name not in ir.tensors:
                shape = (
                    [d.dim_value if d.HasField("dim_value") else d.dim_param for d in vi.type.tensor_type.shape.dim]
                    if vi.type.tensor_type.HasField("shape")
                    else []
                )
                ir.add_tensor(
                    IRTensor(
                        name=vi.name,
                        shape=tuple(shape),
                        dtype=onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type).lower(),
                        is_initializer=False,
                    )
                )

        # Nodes
        from neuron_toolkit.backends.onnx.utils import _node_attrs

        for node in graph_proto.node:
            attrs = _node_attrs(node)
            ir_node = IRNode(
                op_type=node.op_type,
                name=node.name,
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=attrs,
                domain=node.domain,
            )
            ir.add_node(ir_node)

        return ir


class TFLiteAdapter:
    """Adapter for TFLite models."""

    @classmethod
    def can_handle(cls, source: Any) -> bool:
        """Check if source is TFLite bytes or model."""
        if isinstance(source, (bytes, bytearray)):
            return len(source) >= 8 and source[4:8] in (b"TFL3", b"TFL1")
        return False

    def import_to_ir(self, source: Any) -> IRGraph:
        """Convert TFLite model into IRGraph."""
        from neuron_toolkit.backends.tflite.parser import TFLiteParser

        parser = TFLiteParser(source)
        ir = IRGraph(name="tflite_model")

        inputs = sorted(getattr(parser, "inputs", None) or parser.graph_inputs)
        outputs = sorted(getattr(parser, "outputs", None) or parser.graph_outputs)
        for inp in inputs:
            ir.inputs.append(inp)
        for out in outputs:
            ir.outputs.append(out)

        tensor_names = getattr(parser, "_tensor_to_idx", {}).keys()
        for name in tensor_names:
            rank, dtype = parser.shape_info.get(name, (0, "float32"))
            data = parser.tensor_map.get(name) if hasattr(parser, "tensor_map") else None
            shape = tuple(data.shape) if data is not None else tuple([1] * rank)
            ir.add_tensor(
                IRTensor(
                    name=name,
                    shape=shape,
                    dtype=str(dtype),
                    data=data,
                    is_initializer=data is not None,
                )
            )

        for node in parser.nodes:
            ir_node = IRNode(
                op_type=getattr(node, "op_type", ""),
                name=getattr(node, "name", ""),
                inputs=list(getattr(node, "input", [])),
                outputs=list(getattr(node, "output", [])),
                attributes=getattr(node, "attrs", {}),
                domain="tflite",
            )
            ir.add_node(ir_node)

        return ir
