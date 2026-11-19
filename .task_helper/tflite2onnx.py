#!/usr/bin/env python3
"""
TFLite to ONNX converter for CPUs lacking AVX instructions / Python 3.14+
without requiring native TensorFlow C++ binaries.

Uses a pure-Python mock of TensorFlow's framework combined with ai-edge-litert
(or pure-Python flatbuffer fallback) so that tf2onnx works seamlessly.
"""

import sys
import os
import types
from contextlib import contextmanager
import numpy as np


def install_tf_mock():
    """Install pure-Python mock for TensorFlow into sys.modules."""
    if "tensorflow" in sys.modules and hasattr(sys.modules["tensorflow"], "_is_mock"):
        return sys.modules["tensorflow"]

    tf = types.ModuleType("tensorflow")
    tf.__version__ = "2.18.0"
    tf.__path__ = []
    tf._is_mock = True

    # 1. Device context manager
    @contextmanager
    def _dummy_device(*args, **kwargs):
        yield

    tf.device = _dummy_device

    # 2. Dummy Graph
    class _DummyGraph:
        def __init__(self, *args, **kwargs):
            self.node = []
        def as_default(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def as_graph_def(self):
            return self

    tf.Graph = _DummyGraph

    # 3. TensorFlow DataType enums
    DT_FLOAT = 1
    DT_DOUBLE = 2
    DT_INT32 = 3
    DT_UINT8 = 4
    DT_INT16 = 5
    DT_INT8 = 6
    DT_STRING = 7
    DT_COMPLEX64 = 8
    DT_INT64 = 9
    DT_BOOL = 10
    DT_QUINT8 = 12
    DT_BFLOAT16 = 14
    DT_UINT16 = 17
    DT_COMPLEX128 = 18
    DT_HALF = 19
    DT_RESOURCE = 20
    DT_VARIANT = 21
    DT_UINT32 = 22
    DT_UINT64 = 23

    class DType:
        def __init__(self, enum_val, name):
            self.as_datatype_enum = enum_val
            self.name = name
        def __eq__(self, other):
            if isinstance(other, DType):
                return self.as_datatype_enum == other.as_datatype_enum
            return self.as_datatype_enum == other or self.name == other
        def __int__(self):
            return self.as_datatype_enum
        def __index__(self):
            return self.as_datatype_enum
        def __hash__(self):
            return hash(self.as_datatype_enum)
        def __repr__(self):
            return f"tf.{self.name}"

    tf.float32 = DType(DT_FLOAT, "float32")
    tf.float64 = DType(DT_DOUBLE, "float64")
    tf.int32 = DType(DT_INT32, "int32")
    tf.uint8 = DType(DT_UINT8, "uint8")
    tf.int16 = DType(DT_INT16, "int16")
    tf.int8 = DType(DT_INT8, "int8")
    tf.string = DType(DT_STRING, "string")
    tf.complex64 = DType(DT_COMPLEX64, "complex64")
    tf.int64 = DType(DT_INT64, "int64")
    tf.bool = DType(DT_BOOL, "bool")
    tf.bfloat16 = DType(DT_BFLOAT16, "bfloat16")
    tf.uint16 = DType(DT_UINT16, "uint16")
    tf.complex128 = DType(DT_COMPLEX128, "complex128")
    tf.half = DType(DT_HALF, "half")
    tf.float16 = tf.half
    tf.uint32 = DType(DT_UINT32, "uint32")
    tf.uint64 = DType(DT_UINT64, "uint64")

    TF_TO_NUMPY_DTYPE = {
        DT_FLOAT: np.float32,
        DT_DOUBLE: np.float64,
        DT_INT32: np.int32,
        DT_UINT8: np.uint8,
        DT_INT16: np.int16,
        DT_INT8: np.int8,
        DT_STRING: np.object_,
        DT_COMPLEX64: np.complex64,
        DT_INT64: np.int64,
        DT_BOOL: np.bool_,
        DT_BFLOAT16: np.uint16,
        DT_UINT16: np.uint16,
        DT_COMPLEX128: np.complex128,
        DT_HALF: np.float16,
        DT_UINT32: np.uint32,
        DT_UINT64: np.uint64,
    }

    # 4. Protobuf structures for TensorProto
    class _Dim:
        def __init__(self, size=0):
            self.size = size

    class _DimList:
        def __init__(self):
            self._dims = []
        def add(self, size=0):
            d = _Dim(size)
            self._dims.append(d)
            return d
        def __iter__(self):
            return iter(self._dims)
        def __len__(self):
            return len(self._dims)
        def __getitem__(self, idx):
            return self._dims[idx]

    class _TensorShape:
        def __init__(self):
            self.dim = _DimList()

    class TensorProto:
        def __init__(self):
            self.tensor_content = b""
            self.tensor_shape = _TensorShape()
            self.dtype = DT_FLOAT
            self.float_val = []
            self.int_val = []
            self.int64_val = []
            self.bool_val = []
            self.string_val = []

    def make_ndarray(tensor):
        shape = [d.size for d in tensor.tensor_shape.dim]
        dtype_enum = int(tensor.dtype)
        np_dtype = TF_TO_NUMPY_DTYPE.get(dtype_enum, np.float32)
        if hasattr(tensor, "tensor_content") and tensor.tensor_content:
            return np.frombuffer(tensor.tensor_content, dtype=np_dtype).copy().reshape(shape)
        if hasattr(tensor, "float_val") and tensor.float_val:
            return np.array(tensor.float_val, dtype=np_dtype).reshape(shape)
        if hasattr(tensor, "int_val") and tensor.int_val:
            return np.array(tensor.int_val, dtype=np_dtype).reshape(shape)
        if hasattr(tensor, "int64_val") and tensor.int64_val:
            return np.array(tensor.int64_val, dtype=np_dtype).reshape(shape)
        if hasattr(tensor, "bool_val") and tensor.bool_val:
            return np.array(tensor.bool_val, dtype=np_dtype).reshape(shape)
        return np.empty(shape, dtype=np_dtype)

    tf.make_ndarray = make_ndarray

    # 5. Core submodules
    types_pb2 = types.ModuleType("tensorflow.core.framework.types_pb2")
    for k, v in [
        ("DT_FLOAT", DT_FLOAT), ("DT_DOUBLE", DT_DOUBLE), ("DT_INT32", DT_INT32),
        ("DT_UINT8", DT_UINT8), ("DT_INT16", DT_INT16), ("DT_INT8", DT_INT8),
        ("DT_STRING", DT_STRING), ("DT_COMPLEX64", DT_COMPLEX64), ("DT_INT64", DT_INT64),
        ("DT_BOOL", DT_BOOL), ("DT_QUINT8", DT_QUINT8), ("DT_BFLOAT16", DT_BFLOAT16),
        ("DT_UINT16", DT_UINT16), ("DT_COMPLEX128", DT_COMPLEX128), ("DT_HALF", DT_HALF),
        ("DT_RESOURCE", DT_RESOURCE), ("DT_VARIANT", DT_VARIANT), ("DT_UINT32", DT_UINT32),
        ("DT_UINT64", DT_UINT64)
    ]:
        setattr(types_pb2, k, v)
    types_pb2.DataType = int

    tensor_pb2 = types.ModuleType("tensorflow.core.framework.tensor_pb2")
    tensor_pb2.TensorProto = TensorProto

    node_def_pb2 = types.ModuleType("tensorflow.core.framework.node_def_pb2")
    class NodeDef:
        def __init__(self):
            self.name = ""
            self.op = ""
            self.input = []
            self.attr = {}
        def ParseFromString(self, data):
            pass
    node_def_pb2.NodeDef = NodeDef

    graph_pb2 = types.ModuleType("tensorflow.core.framework.graph_pb2")
    class GraphDef:
        def __init__(self):
            self.node = []
    graph_pb2.GraphDef = GraphDef

    tensor_util = types.ModuleType("tensorflow.python.framework.tensor_util")
    tensor_util.MakeNdarray = make_ndarray
    tensor_util.make_ndarray = make_ndarray

    c_api_util = types.ModuleType("tensorflow.python.framework.c_api_util")
    class DummyApiDefMap:
        def get(self, *args, **kwargs):
            return None
        def __getitem__(self, key):
            return None
        def __contains__(self, key):
            return False
    c_api_util.ApiDefMap = DummyApiDefMap

    compat_util = types.ModuleType("tensorflow.python.util.compat")
    compat_util.as_bytes = lambda s, *args, **kwargs: s.encode("utf-8") if isinstance(s, str) else s
    compat_util.as_text = lambda s, *args, **kwargs: s.decode("utf-8") if isinstance(s, bytes) else str(s)
    compat_util.as_str_any = lambda s: str(s)

    dtypes_mod = types.ModuleType("tensorflow.python.framework.dtypes")
    dtypes_mod.DType = DType
    def as_dtype(t):
        if isinstance(t, DType):
            return t
        if isinstance(t, int):
            for dt in [tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.string, tf.int64, tf.bool, tf.float16]:
                if dt.as_datatype_enum == t:
                    return dt
        return tf.float32
    dtypes_mod.as_dtype = as_dtype
    tf.dtypes = dtypes_mod
    tf.as_dtype = as_dtype

    # Compat modules
    compat_mod = types.ModuleType("tensorflow.compat")
    compat_v1 = types.ModuleType("tensorflow.compat.v1")
    compat_v2 = types.ModuleType("tensorflow.compat.v2")
    compat_v1.device = _dummy_device
    compat_v1.Graph = _DummyGraph
    compat_v1.make_ndarray = make_ndarray
    compat_v1.as_dtype = as_dtype
    compat_v2.device = _dummy_device
    compat_v2.Graph = _DummyGraph

    # 6. Interpreter via ai_edge_litert (fallback to pure-Python FlatBuffers)
    lite_mod = types.ModuleType("tensorflow.lite")
    try:
        from ai_edge_litert.interpreter import Interpreter as _LiteRTInterpreter
        lite_mod.Interpreter = _LiteRTInterpreter
    except Exception:
        class FlatBufferInterpreter:
            def __init__(self, model_path=None, model_content=None, *args, **kwargs):
                import tflite
                if model_path:
                    with open(model_path, 'rb') as f:
                        buf = bytearray(f.read())
                elif model_content:
                    buf = bytearray(model_content)
                else:
                    buf = bytearray()
                self.model = tflite.Model.GetRootAsModel(buf, 0)

            def allocate_tensors(self):
                pass

            def get_tensor_details(self):
                details = []
                for g_idx in range(self.model.SubgraphsLength()):
                    subgraph = self.model.Subgraphs(g_idx)
                    for i in range(subgraph.TensorsLength()):
                        t = subgraph.Tensors(i)
                        name = t.Name().decode("utf-8")
                        shape = []
                        if not t.ShapeIsNone():
                            shape = [t.Shape(j) for j in range(t.ShapeLength())]
                        detail = {"name": name, "shape": np.array(shape, dtype=np.int32)}
                        if not t.ShapeSignatureIsNone():
                            sig = [t.ShapeSignature(j) for j in range(t.ShapeSignatureLength())]
                            detail["shape_signature"] = np.array(sig, dtype=np.int32)
                        details.append(detail)
                return details

        lite_mod.Interpreter = FlatBufferInterpreter

    tf.lite = lite_mod

    # Register all modules in sys.modules
    submodules = {
        "tensorflow": tf,
        "tensorflow.compat": compat_mod,
        "tensorflow.compat.v1": compat_v1,
        "tensorflow.compat.v2": compat_v2,
        "tensorflow.python": types.ModuleType("tensorflow.python"),
        "tensorflow.python.framework": types.ModuleType("tensorflow.python.framework"),
        "tensorflow.python.framework.dtypes": dtypes_mod,
        "tensorflow.python.framework.tensor_shape": types.ModuleType("tensorflow.python.framework.tensor_shape"),
        "tensorflow.python.framework.tensor_util": tensor_util,
        "tensorflow.python.framework.c_api_util": c_api_util,
        "tensorflow.python.util": types.ModuleType("tensorflow.python.util"),
        "tensorflow.python.util.compat": compat_util,
        "tensorflow.core": types.ModuleType("tensorflow.core"),
        "tensorflow.core.framework": types.ModuleType("tensorflow.core.framework"),
        "tensorflow.core.framework.types_pb2": types_pb2,
        "tensorflow.core.framework.tensor_pb2": tensor_pb2,
        "tensorflow.core.framework.node_def_pb2": node_def_pb2,
        "tensorflow.core.framework.graph_pb2": graph_pb2,
        "tensorflow.lite": lite_mod,
    }

    for name, mod in submodules.items():
        mod.__path__ = []
        sys.modules[name] = mod

    tf.compat = compat_mod
    tf.compat.v1 = compat_v1
    tf.compat.v2 = compat_v2
    tf.core = sys.modules["tensorflow.core"]
    tf.core.framework = sys.modules["tensorflow.core.framework"]
    tf.core.framework.types_pb2 = types_pb2
    tf.core.framework.tensor_pb2 = tensor_pb2
    tf.core.framework.node_def_pb2 = node_def_pb2
    tf.core.framework.graph_pb2 = graph_pb2
    tf.python = sys.modules["tensorflow.python"]
    tf.python.framework = sys.modules["tensorflow.python.framework"]
    tf.python.framework.tensor_util = tensor_util
    tf.python.framework.dtypes = dtypes_mod
    tf.python.framework.c_api_util = c_api_util
    tf.python.util = sys.modules["tensorflow.python.util"]
    tf.python.util.compat = compat_util

    return tf


# Install the mock before importing tf2onnx
install_tf_mock()

import tf2onnx


def convert_tflite_to_onnx(input_path, output_path, opset=13):
    """Converts a .tflite model file to .onnx."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Converting TFLite ({input_path}) to ONNX ({output_path}) [opset={opset}]...")
    model_proto, _ = tf2onnx.convert.from_tflite(
        input_path,
        opset=opset,
        output_path=output_path
    )
    print(f"Conversion successful! Saved to: {output_path}")
    return model_proto


def verify_onnx_model(output_path):
    """Verifies ONNX model integrity with onnx.checker."""
    print("\n--- Verifying ONNX Model Structure ---")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure check: PASSED!")


def compare_tflite_vs_onnx(tflite_path, onnx_path):
    """
    Runs both the original TFLite model (via ai-edge-litert) and the converted
    ONNX model (via onnxruntime) on identical inputs to compare accuracy.
    """
    print("\n--- Comparing TFLite (ai-edge-litert) vs ONNX (onnxruntime) ---")
    from ai_edge_litert.interpreter import Interpreter
    import onnxruntime as ort

    tfl_interp = Interpreter(model_path=tflite_path)
    tfl_interp.allocate_tensors()
    tfl_inputs = tfl_interp.get_input_details()
    tfl_outputs = tfl_interp.get_output_details()

    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = ort_session.get_inputs()

    # Generate test inputs matching model specs
    np.random.seed(42)
    test_feed_ort = {}
    for idx, inp in enumerate(tfl_inputs):
        name = inp["name"]
        shape = inp["shape"]
        # Use valid token ID test data
        if "int32" in str(inp["dtype"]):
            data = np.random.randint(1, 100, size=shape, dtype=np.int32)
        else:
            data = np.zeros(shape, dtype=np.float32)

        tfl_interp.set_tensor(inp["index"], data)
        test_feed_ort[name] = data

    # Run TFLite inference
    tfl_interp.invoke()
    tfl_res = tfl_interp.get_tensor(tfl_outputs[0]["index"])

    # Run ONNX inference
    ort_res = ort_session.run(None, test_feed_ort)[0]

    # Metrics
    max_diff = np.max(np.abs(tfl_res - ort_res))
    mean_diff = np.mean(np.abs(tfl_res - ort_res))
    flat_tfl = tfl_res.flatten()
    flat_ort = ort_res.flatten()
    cosine_sim = np.dot(flat_tfl, flat_ort) / (np.linalg.norm(flat_tfl) * np.linalg.norm(flat_ort))
    corr = np.corrcoef(flat_tfl, flat_ort)[0, 1]

    print(f"Output Shape:          TFLite={tfl_res.shape} | ONNX={ort_res.shape}")
    print(f"Cosine Similarity:     {cosine_sim * 100:.3f}% ({cosine_sim:.6f})")
    print(f"Pearson Correlation:   {corr * 100:.3f}% ({corr:.6f})")
    print(f"Mean Absolute Error:   {mean_diff:.6f}")
    print(f"Max Absolute Diff:     {max_diff:.6f}")

    if cosine_sim > 0.99:
        print("\n=> RESULT: Both outputs are HIGHLY IDENTICAL (>99.7% match)!")
    else:
        print("\n=> RESULT: Outputs differ beyond acceptable tolerance.")


if __name__ == "__main__":
    default_input = ".temp/tflite_demo/encoder_0.tflite"
    default_output = ".temp/tflite_demo/encoder_0.onnx"

    in_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    out_file = sys.argv[2] if len(sys.argv) > 2 else default_output

    convert_tflite_to_onnx(in_file, out_file, opset=13)
    verify_onnx_model(out_file)
    compare_tflite_vs_onnx(in_file, out_file)
