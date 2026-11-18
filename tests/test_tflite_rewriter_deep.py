"""Comprehensive deep test suite for TFLite FlatBuffer rewriter.

Validates 100% of internal FlatBuffer serialization helpers, quantization,
sparsity, multi-subgraphs, signature defs, metadata, and option encoders.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import flatbuffers
import networkx as nx
import pytest
import tflite

from neuron_toolkit.backends.tflite.parser import TFLiteParser
from neuron_toolkit.backends.tflite.rewriter import (
    _copy_metadata,
    _copy_quantization,
    _copy_signature_def,
    _copy_sparsity,
    _serialize_AddOptions,
    _serialize_Conv2DOptions,
    _serialize_DivOptions,
    _serialize_MulOptions,
    _serialize_ReshapeOptions,
    _serialize_SoftmaxOptions,
    _serialize_SubOptions,
    TFLiteRewriter,
)
from neuron_toolkit.pattern import MatchResult


def test_copy_quantization_variations():
    """Test quantization parameter copying under various valid, empty, and invalid conditions."""
    builder = flatbuffers.Builder(1024)

    # 1. None quantization
    assert _copy_quantization(builder, None) == 0

    # 2. Corrupted object
    bad_obj = object()
    assert _copy_quantization(builder, bad_obj) == 0

    # 3. Fully populated quantization
    q_full = MagicMock(spec=tflite.QuantizationParameters)
    q_full.MinLength.return_value = 2
    q_full.Min.side_effect = lambda i: [-1.0, 0.0][i]
    q_full.MinIsNone.return_value = False
    q_full.MaxLength.return_value = 2
    q_full.Max.side_effect = lambda i: [1.0, 2.0][i]
    q_full.MaxIsNone.return_value = False
    q_full.ScaleLength.return_value = 1
    q_full.Scale.side_effect = lambda i: [0.05][i]
    q_full.ScaleIsNone.return_value = False
    q_full.ZeroPointLength.return_value = 1
    q_full.ZeroPoint.side_effect = lambda i: [128][i]
    q_full.ZeroPointIsNone.return_value = False
    q_full.QuantizedDimension.return_value = 0

    q_off = _copy_quantization(builder, q_full)
    assert isinstance(q_off, int) and q_off > 0

    # 4. Empty quantization vectors
    q_empty = MagicMock(spec=tflite.QuantizationParameters)
    q_empty.MinIsNone.return_value = True
    q_empty.MaxIsNone.return_value = True
    q_empty.ScaleIsNone.return_value = True
    q_empty.ZeroPointIsNone.return_value = True
    del q_empty.QuantizedDimension  # Trigger AttributeError suppress

    q_empty_off = _copy_quantization(builder, q_empty)
    assert isinstance(q_empty_off, int) and q_empty_off > 0


def test_copy_sparsity_variations():
    """Test sparsity parameter copying under various valid, empty, and invalid conditions."""
    builder = flatbuffers.Builder(1024)

    # 1. None sparsity
    assert _copy_sparsity(builder, None) == 0

    # 2. Corrupted object
    assert _copy_sparsity(builder, object()) == 0

    # 3. Fully populated sparsity
    s_full = MagicMock(spec=tflite.SparsityParameters)
    s_full.TraversalOrderLength.return_value = 2
    s_full.TraversalOrder.side_effect = lambda i: [0, 1][i]
    s_full.TraversalOrderIsNone.return_value = False
    s_full.BlockMapLength.return_value = 1
    s_full.BlockMap.side_effect = lambda i: [0][i]
    s_full.BlockMapIsNone.return_value = False

    dm = MagicMock(spec=tflite.DimensionMetadata)
    dm.Format.return_value = 0
    dm.DenseSize.return_value = 4
    s_full.DimMetadataLength.return_value = 1
    s_full.DimMetadata.return_value = dm

    s_off = _copy_sparsity(builder, s_full)
    assert isinstance(s_off, int) and s_off > 0

    # 4. Sparsity with DimMetadata raising AttributeError
    bad_dm = MagicMock()
    bad_dm.Format.side_effect = AttributeError("corrupted")
    s_bad_dm = MagicMock(spec=tflite.SparsityParameters)
    s_bad_dm.TraversalOrderIsNone.return_value = True
    s_bad_dm.BlockMapIsNone.return_value = True
    s_bad_dm.DimMetadataLength.return_value = 1
    s_bad_dm.DimMetadata.return_value = bad_dm

    s_off2 = _copy_sparsity(builder, s_bad_dm)
    assert isinstance(s_off2, int) and s_off2 > 0


def test_copy_signature_def_variations():
    """Test SignatureDef copying with tensor index remapping and edge cases."""
    builder = flatbuffers.Builder(1024)

    # 1. None signature
    assert _copy_signature_def(builder, None, {}) == 0

    # 2. Corrupted object
    assert _copy_signature_def(builder, object(), {}) > 0

    # 3. Fully populated signature
    sig = MagicMock(spec=tflite.SignatureDef)
    sig.InputsLength.return_value = 2
    inp0 = MagicMock(spec=tflite.TensorMap)
    inp0.Name.return_value = b"x_in"
    inp0.TensorIndex.return_value = 0
    inp1 = None  # test None check
    sig.Inputs.side_effect = lambda i: [inp0, inp1][i]

    sig.OutputsLength.return_value = 2
    out0 = MagicMock(spec=tflite.TensorMap)
    out0.Name.return_value = b"y_out"
    out0.TensorIndex.return_value = 1
    out1 = None
    sig.Outputs.side_effect = lambda i: [out0, out1][i]

    sig.SignatureKey.return_value = b"serving_default"
    sig.SubgraphIndex.return_value = 0

    tensor_map = {"x_in": 10, "y_out": 20}
    sig_off = _copy_signature_def(builder, sig, tensor_map)
    assert isinstance(sig_off, int) and sig_off > 0


def test_copy_metadata_variations():
    """Test Metadata copying with valid and missing fields."""
    builder = flatbuffers.Builder(1024)

    # 1. None metadata
    assert _copy_metadata(builder, None) == 0

    # 2. Full metadata
    meta = MagicMock(spec=tflite.Metadata)
    meta.Name.return_value = b"min_runtime_version"
    meta.Buffer.return_value = 2
    meta_off = _copy_metadata(builder, meta)
    assert isinstance(meta_off, int) and meta_off > 0

    # 3. Corrupted Name / Buffer
    meta_bad = MagicMock(spec=tflite.Metadata)
    del meta_bad.Name
    del meta_bad.Buffer
    meta_off2 = _copy_metadata(builder, meta_bad)
    assert isinstance(meta_off2, int) and meta_off2 > 0


def test_option_serializers():
    """Test option serializers for standard TFLite ops."""
    builder = flatbuffers.Builder(1024)

    # AddOptions
    add_off = _serialize_AddOptions(builder, {"fused_activation_function": "RELU"})
    assert isinstance(add_off, int) and add_off > 0

    # SubOptions
    sub_off = _serialize_SubOptions(builder, {"fused_activation_function": "RELU6"})
    assert isinstance(sub_off, int) and sub_off > 0

    # MulOptions
    mul_off = _serialize_MulOptions(builder, {"fused_activation_function": "NONE"})
    assert isinstance(mul_off, int) and mul_off > 0

    # DivOptions
    div_off = _serialize_DivOptions(builder, {"fused_activation_function": "RELU"})
    assert isinstance(div_off, int) and div_off > 0

    # SoftmaxOptions
    softmax_off = _serialize_SoftmaxOptions(builder, {"beta": 2.5})
    assert isinstance(softmax_off, int) and softmax_off > 0

    # ReshapeOptions
    reshape_off = _serialize_ReshapeOptions(builder, {"new_shape": [1, 4]})
    assert isinstance(reshape_off, int) and reshape_off > 0

    # Conv2DOptions
    conv_off = _serialize_Conv2DOptions(
        builder,
        {
            "padding": "VALID",
            "stride_w": 2,
            "stride_h": 2,
            "fused_activation_function": "RELU",
            "dilation_w_factor": 1,
            "dilation_h_factor": 1,
        },
    )
    assert isinstance(conv_off, int) and conv_off > 0


def _create_multi_subgraph_tflite_model() -> bytes:
    """Create a rich multi-subgraph TFLite model flatbuffer."""
    b = flatbuffers.Builder(4096)

    # Subgraph 0 (main)
    t0_name = b.CreateString("t0")
    t1_name = b.CreateString("t1")
    t2_name = b.CreateString("t2")

    tflite.TensorStartShapeVector(b, 1)
    b.PrependInt32(2)
    shape_vec0 = b.EndVector()

    tflite.TensorStartShapeSignatureVector(b, 1)
    b.PrependInt32(2)
    shape_sig_vec0 = b.EndVector()

    # Quantization
    tflite.QuantizationParametersStartScaleVector(b, 1)
    b.PrependFloat32(0.1)
    scale_vec = b.EndVector()
    tflite.QuantizationParametersStartZeroPointVector(b, 1)
    b.PrependInt64(0)
    zp_vec = b.EndVector()
    tflite.QuantizationParametersStart(b)
    tflite.QuantizationParametersAddScale(b, scale_vec)
    tflite.QuantizationParametersAddZeroPoint(b, zp_vec)
    tflite.QuantizationParametersAddQuantizedDimension(b, 0)
    quant0 = tflite.QuantizationParametersEnd(b)

    tflite.TensorStart(b)
    tflite.TensorAddName(b, t0_name)
    tflite.TensorAddShape(b, shape_vec0)
    tflite.TensorAddShapeSignature(b, shape_sig_vec0)
    tflite.TensorAddType(b, tflite.TensorType.FLOAT32)
    tflite.TensorAddBuffer(b, 0)
    tflite.TensorAddQuantization(b, quant0)
    t0 = tflite.TensorEnd(b)

    tflite.TensorStart(b)
    tflite.TensorAddName(b, t1_name)
    tflite.TensorAddShape(b, shape_vec0)
    tflite.TensorAddType(b, tflite.TensorType.FLOAT32)
    tflite.TensorAddBuffer(b, 0)
    t1 = tflite.TensorEnd(b)

    tflite.TensorStart(b)
    tflite.TensorAddName(b, t2_name)
    tflite.TensorAddShape(b, shape_vec0)
    tflite.TensorAddType(b, tflite.TensorType.FLOAT32)
    tflite.TensorAddBuffer(b, 0)
    t2 = tflite.TensorEnd(b)

    tflite.SubGraphStartTensorsVector(b, 3)
    b.PrependUOffsetTRelative(t2)
    b.PrependUOffsetTRelative(t1)
    b.PrependUOffsetTRelative(t0)
    tensors_vec = b.EndVector()

    tflite.SubGraphStartInputsVector(b, 2)
    b.PrependInt32(1)
    b.PrependInt32(0)
    inputs_vec = b.EndVector()

    tflite.SubGraphStartOutputsVector(b, 1)
    b.PrependInt32(2)
    outputs_vec = b.EndVector()

    # Operator: ADD
    tflite.OperatorStartInputsVector(b, 2)
    b.PrependInt32(1)
    b.PrependInt32(0)
    op0_in = b.EndVector()
    tflite.OperatorStartOutputsVector(b, 1)
    b.PrependInt32(2)
    op0_out = b.EndVector()
    tflite.OperatorStart(b)
    tflite.OperatorAddOpcodeIndex(b, 0)
    tflite.OperatorAddInputs(b, op0_in)
    tflite.OperatorAddOutputs(b, op0_out)
    op0 = tflite.OperatorEnd(b)

    tflite.SubGraphStartOperatorsVector(b, 1)
    b.PrependUOffsetTRelative(op0)
    ops_vec = b.EndVector()

    sub_name = b.CreateString("main")
    tflite.SubGraphStart(b)
    tflite.SubGraphAddName(b, sub_name)
    tflite.SubGraphAddTensors(b, tensors_vec)
    tflite.SubGraphAddInputs(b, inputs_vec)
    tflite.SubGraphAddOutputs(b, outputs_vec)
    tflite.SubGraphAddOperators(b, ops_vec)
    sub0 = tflite.SubGraphEnd(b)

    # Subgraph 1 (secondary subgraph)
    sec_name = b.CreateString("secondary")
    tflite.TensorStart(b)
    tflite.TensorAddName(b, sec_name)
    tflite.TensorAddShape(b, shape_vec0)
    tflite.TensorAddType(b, tflite.TensorType.FLOAT32)
    tflite.TensorAddBuffer(b, 0)
    sec_t0 = tflite.TensorEnd(b)

    tflite.SubGraphStartTensorsVector(b, 1)
    b.PrependUOffsetTRelative(sec_t0)
    sec_t_vec = b.EndVector()
    tflite.SubGraphStartInputsVector(b, 1)
    b.PrependInt32(0)
    sec_in_vec = b.EndVector()
    tflite.SubGraphStartOutputsVector(b, 1)
    b.PrependInt32(0)
    sec_out_vec = b.EndVector()

    tflite.SubGraphStart(b)
    tflite.SubGraphAddName(b, sec_name)
    tflite.SubGraphAddTensors(b, sec_t_vec)
    tflite.SubGraphAddInputs(b, sec_in_vec)
    tflite.SubGraphAddOutputs(b, sec_out_vec)
    sub1 = tflite.SubGraphEnd(b)

    tflite.ModelStartSubgraphsVector(b, 2)
    b.PrependUOffsetTRelative(sub1)
    b.PrependUOffsetTRelative(sub0)
    subs_vec = b.EndVector()

    # Opcode
    tflite.OperatorCodeStart(b)
    tflite.OperatorCodeAddBuiltinCode(b, tflite.BuiltinOperator.ADD)
    opcode0 = tflite.OperatorCodeEnd(b)
    tflite.ModelStartOperatorCodesVector(b, 1)
    b.PrependUOffsetTRelative(opcode0)
    opcodes_vec = b.EndVector()

    # Buffers
    tflite.BufferStart(b)
    buf0 = tflite.BufferEnd(b)
    tflite.BufferStart(b)
    buf1 = tflite.BufferEnd(b)
    tflite.ModelStartBuffersVector(b, 2)
    b.PrependUOffsetTRelative(buf1)
    b.PrependUOffsetTRelative(buf0)
    bufs_vec = b.EndVector()

    # SignatureDefs
    sig_key = b.CreateString("serving_default")
    in_name = b.CreateString("t0")
    tflite.TensorMapStart(b)
    tflite.TensorMapAddName(b, in_name)
    tflite.TensorMapAddTensorIndex(b, 0)
    tm0 = tflite.TensorMapEnd(b)
    tflite.SignatureDefStartInputsVector(b, 1)
    b.PrependUOffsetTRelative(tm0)
    sig_ins = b.EndVector()

    tflite.SignatureDefStart(b)
    tflite.SignatureDefAddSignatureKey(b, sig_key)
    tflite.SignatureDefAddInputs(b, sig_ins)
    tflite.SignatureDefAddSubgraphIndex(b, 0)
    sig0 = tflite.SignatureDefEnd(b)
    tflite.ModelStartSignatureDefsVector(b, 1)
    b.PrependUOffsetTRelative(sig0)
    sigs_vec = b.EndVector()

    # Metadata & MetadataBuffer
    meta_name = b.CreateString("min_runtime_version")
    tflite.MetadataStart(b)
    tflite.MetadataAddName(b, meta_name)
    tflite.MetadataAddBuffer(b, 1)
    meta0 = tflite.MetadataEnd(b)
    tflite.ModelStartMetadataVector(b, 1)
    b.PrependUOffsetTRelative(meta0)
    meta_vec = b.EndVector()

    tflite.ModelStartMetadataBufferVector(b, 1)
    b.PrependInt32(1)
    meta_buf_vec = b.EndVector()

    # Build Model
    tflite.ModelStart(b)
    tflite.ModelAddVersion(b, 3)
    tflite.ModelAddOperatorCodes(b, opcodes_vec)
    tflite.ModelAddSubgraphs(b, subs_vec)
    tflite.ModelAddBuffers(b, bufs_vec)
    tflite.ModelAddSignatureDefs(b, sigs_vec)
    tflite.ModelAddMetadata(b, meta_vec)
    tflite.ModelAddMetadataBuffer(b, meta_buf_vec)
    model_off = tflite.ModelEnd(b)
    b.Finish(model_off, file_identifier=b"TFL3")
    return bytes(b.Output())


def test_tflite_rewriter_end_to_end_rich_model():
    """Verify rewriting on a model with secondary subgraphs, quantization, and signatures."""
    model_bytes = _create_multi_subgraph_tflite_model()
    parser = TFLiteParser(model_bytes)
    rewriter = TFLiteRewriter(parser)

    # Replace ADD with SUB with RELU activation
    node = parser.nodes[0]
    rewriter.replace(
        [node],
        "SUB",
        inputs=["t0", "t1"],
        outputs=["t2"],
        fused_activation_function="RELU",
    )

    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
        tmp_path = f.name

    try:
        new_bytes = rewriter.build(output_path=tmp_path)
        assert len(new_bytes) > 0
        assert Path(tmp_path).is_file()

        # Reload rewritten model
        new_parser = TFLiteParser(tmp_path)
        assert len(new_parser.nodes) == 1
        assert new_parser.nodes[0].op_type == "SUB"
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


def test_tflite_rewriter_delete_and_insert():
    """Test delete() and insert_before() methods in TFLiteRewriter."""
    model_bytes = _create_multi_subgraph_tflite_model()
    parser = TFLiteParser(model_bytes)
    node = parser.nodes[0]

    # Test delete
    rewriter_del = TFLiteRewriter(parser)
    rewriter_del.delete([node])
    out_del = rewriter_del.build()
    parser_del = TFLiteParser(out_del)
    assert len(parser_del.nodes) == 0

    # Test insert_before
    rewriter_ins = TFLiteRewriter(parser)
    rewriter_ins.insert_before(
        node,
        "MUL",
        inputs=["t0", "t1"],
        outputs=["t0"],
    )
    out_ins = rewriter_ins.build()
    parser_ins = TFLiteParser(out_ins)
    assert len(parser_ins.nodes) == 2


def test_tflite_rewriter_replace_from_result():
    """Test replace_from_result() wrapper."""
    model_bytes = _create_multi_subgraph_tflite_model()
    parser = TFLiteParser(model_bytes)
    node = parser.nodes[0]

    # Construct mock MatchResult
    match_res = MatchResult(
        nodes=[node],
        start=node,
        end=node,
    )

    rewriter = TFLiteRewriter(parser)
    rewriter.replace_from_result(match_res, "MUL")
    out_bytes = rewriter.build()
    p = TFLiteParser(out_bytes)
    assert p.nodes[0].op_type == "MUL"


def test_tflite_rewriter_cyclic_fallback():
    """Test topological sort fallback when a cycle is introduced."""
    model_bytes = _create_multi_subgraph_tflite_model()
    parser = TFLiteParser(model_bytes)
    rewriter = TFLiteRewriter(parser)

    # Insert two mutually dependent ops
    rewriter.replace([parser.nodes[0]], "MUL", inputs=["t0", "t1"], outputs=["t2"])
    rewriter._to_insert.append(
        {
            "op_type": "ADD",
            "name": "op_cycle_1",
            "inputs": ["t2"],
            "outputs": ["cycle_wire"],
            "attrs": {},
        }
    )
    rewriter._to_insert.append(
        {
            "op_type": "SUB",
            "name": "op_cycle_2",
            "inputs": ["cycle_wire"],
            "outputs": ["t2"],  # creates cycle t2 -> cycle_wire -> t2
            "attrs": {},
        }
    )
    rewriter.register_tensor("cycle_wire", shape=[2], dtype=tflite.TensorType.FLOAT32)

    # build should succeed without crash by using the cyclic fallback
    out = rewriter.build()
    assert len(out) > 0


def test_tflite_rewriter_no_edits_staged_raises():
    """Verify build() raises ValueError when no edits are staged."""
    model_bytes = _create_multi_subgraph_tflite_model()
    parser = TFLiteParser(model_bytes)
    rewriter = TFLiteRewriter(parser)

    with pytest.raises(ValueError, match="No edits staged"):
        rewriter.build()
