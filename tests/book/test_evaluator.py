"""Tests for ``dlshogi.book.evaluator`` (task 10.5, the onnxruntime adapter).

Covers the parts of `OnnxRuntimeInferenceSession` that do not require a
GPU, TensorRT, or CUDA to be present:

- the zero-pad-a-short-batch-then-slice-back-down logic in `run`, tested
  against a stub `_run_iobinding` so no onnxruntime session is involved at
  all (`test_run_pads_a_short_batch_...`, `test_run_does_not_pad_...`);
- constructing a real ``onnxruntime.InferenceSession`` against a tiny
  synthetic ``.onnx`` model with the exact `input1`/`input2`/
  `output_policy`/`output_value` names and dynamic batch axis
  `dlshogi/convert_model_to_onnx.py` produces, and calling `run` on it
  end-to-end through `io_binding` (`test_adapter_loads_and_runs_...`).

**What this file does NOT verify (stated per this task's own
instructions).** This development machine has no GPU (confirmed:
``onnxruntime.get_available_providers()`` returns only
``['AzureExecutionProvider', 'CPUExecutionProvider']``), so:

- the ``TensorrtExecutionProvider`` / ``CUDAExecutionProvider`` code paths
  themselves (engine building, `trt_fp16_enable`, `trt_engine_cache_enable`,
  actual GPU inference) are exercised only as *requested* provider names
  in the list passed to ``ort.InferenceSession`` -- onnxruntime's own
  fallback-to-CPU behavior is what is actually observed here, which is
  also the fallback path task 10.5 asks the adapter to support;
- there is no way, on this machine, to distinguish "TensorRT requested and
  correctly falls back to CUDA" from "TensorRT requested and correctly
  falls back all the way to CPU" from a real GPU's execution -- both this
  module's `test_adapter_resolves_a_provider_and_logs_it` and the
  provider-selection logic it exercises would need to run on GPU hardware
  with the ``onnxruntime-gpu`` package's TensorRT/CUDA shared libraries
  installed to verify the actual TensorRT-first behavior end-to-end;
- the actual zero-padding avoiding a *TensorRT engine rebuild* (the
  reason design.md asks for padding at all) cannot be observed without a
  TensorRT engine to rebuild; what is verified here is only the
  input/output array shapes the padding logic produces, which is the
  entire externally-visible contract `Evaluator._dispatch` depends on.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from dlshogi.book.evaluator import FEATURES1_NUM, FEATURES2_NUM, OnnxRuntimeInferenceSession

POLICY_DIM = 9 * 9 * 27  # 2187, per design.md and dlshogi.common.MAX_MOVE_LABEL_NUM * 81


def _make_tiny_book_model(path) -> None:
    """Write a tiny, valid ``.onnx`` model with the Evaluator's exact I/O contract.

    Mirrors ``dlshogi/convert_model_to_onnx.py``'s ``input1``/``input2``/
    ``output_policy``/``output_value`` names and dynamic ``batch_size``
    axis on every one of those four tensors. The computation itself is
    deliberately trivial (global-average-pool each input, then a
    zero-weight matmul into the right output width) -- this model exists
    only to exercise `OnnxRuntimeInferenceSession`'s session construction,
    provider resolution, and `io_binding` plumbing, not to produce
    meaningful policy/value numbers.
    """
    input1 = helper.make_tensor_value_info(
        "input1", TensorProto.FLOAT, ["batch_size", FEATURES1_NUM, 9, 9]
    )
    input2 = helper.make_tensor_value_info(
        "input2", TensorProto.FLOAT, ["batch_size", FEATURES2_NUM, 9, 9]
    )
    output_policy = helper.make_tensor_value_info(
        "output_policy", TensorProto.FLOAT, ["batch_size", POLICY_DIM]
    )
    output_value = helper.make_tensor_value_info(
        "output_value", TensorProto.FLOAT, ["batch_size", 1]
    )

    w_policy = helper.make_tensor(
        "w_policy", TensorProto.FLOAT, [FEATURES1_NUM, POLICY_DIM],
        np.zeros(FEATURES1_NUM * POLICY_DIM, dtype=np.float32),
    )
    w_value = helper.make_tensor(
        "w_value", TensorProto.FLOAT, [FEATURES2_NUM, 1],
        np.zeros(FEATURES2_NUM, dtype=np.float32),
    )

    nodes = [
        helper.make_node("GlobalAveragePool", ["input1"], ["pooled1"]),
        helper.make_node("Flatten", ["pooled1"], ["flat1"], axis=1),
        helper.make_node("MatMul", ["flat1", "w_policy"], ["output_policy"]),
        helper.make_node("GlobalAveragePool", ["input2"], ["pooled2"]),
        helper.make_node("Flatten", ["pooled2"], ["flat2"], axis=1),
        helper.make_node("MatMul", ["flat2", "w_value_pre"], ["value_logit"]),
        helper.make_node("Sigmoid", ["value_logit"], ["output_value"]),
    ]
    # w_value is named w_value_pre in the node list above but declared as
    # w_value in the initializer; fix the initializer name to match.
    nodes[5] = helper.make_node("MatMul", ["flat2", "w_value"], ["value_logit"])

    graph = helper.make_graph(
        nodes,
        "tiny_book_model",
        [input1, input2],
        [output_policy, output_value],
        initializer=[w_policy, w_value],
    )
    model = helper.make_model(graph, producer_name="test_evaluator")
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture(scope="module")
def tiny_book_model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("evaluator_onnx") / "tiny_book_model.onnx"
    _make_tiny_book_model(path)
    return path


# ---------------------------------------------------------------------------
# Zero-pad / slice-back logic, with no onnxruntime session involved at all.
# ---------------------------------------------------------------------------


def _make_unconstructed_adapter(fixed_batch_size):
    """An `OnnxRuntimeInferenceSession` with `__init__` skipped entirely.

    Tests the padding/slicing logic in `run` in isolation: no
    ``onnxruntime.InferenceSession`` is constructed, so this works on any
    machine regardless of what providers are available. `_run_iobinding`
    is monkeypatched separately by each test below.
    """
    adapter = object.__new__(OnnxRuntimeInferenceSession)
    adapter._fixed_batch_size = fixed_batch_size
    return adapter


def test_run_pads_a_short_batch_up_to_fixed_batch_size_and_slices_output_back_down():
    adapter = _make_unconstructed_adapter(fixed_batch_size=8)
    seen_shapes = {}

    def fake_run_iobinding(features1, features2):
        seen_shapes["f1"] = features1.shape
        seen_shapes["f2"] = features2.shape
        # Return arrays shaped for the padded batch, as a real session
        # would; `run` must slice these back down to n rows before
        # returning to the caller.
        n_padded = features1.shape[0]
        policy = np.arange(n_padded * POLICY_DIM, dtype=np.float32).reshape(
            n_padded, POLICY_DIM
        )
        value = np.arange(n_padded, dtype=np.float32).reshape(n_padded, 1)
        return policy, value

    adapter._run_iobinding = fake_run_iobinding

    n = 3
    f1 = np.random.default_rng(0).standard_normal((n, FEATURES1_NUM, 9, 9)).astype(np.float32)
    f2 = np.random.default_rng(1).standard_normal((n, FEATURES2_NUM, 9, 9)).astype(np.float32)

    policy, value = adapter.run(f1, f2)

    # The underlying session was invoked with the padded (fixed) batch size.
    assert seen_shapes["f1"] == (8, FEATURES1_NUM, 9, 9)
    assert seen_shapes["f2"] == (8, FEATURES2_NUM, 9, 9)
    # The caller sees results shaped for exactly the n=3 rows it asked about.
    assert policy.shape == (n, POLICY_DIM)
    assert value.shape == (n, 1)
    # And those are the first n rows of what the (padded) session returned,
    # i.e. the padding rows were discarded rather than merely truncated
    # from the wrong end or reordered.
    np.testing.assert_array_equal(policy, np.arange(n * POLICY_DIM, dtype=np.float32).reshape(n, POLICY_DIM))
    np.testing.assert_array_equal(value, np.arange(n, dtype=np.float32).reshape(n, 1))


def test_run_does_not_pad_when_batch_already_at_fixed_size():
    adapter = _make_unconstructed_adapter(fixed_batch_size=4)
    seen_shapes = {}

    def fake_run_iobinding(features1, features2):
        seen_shapes["f1"] = features1.shape
        return features1[:, 0, 0, 0].reshape(-1, 1).repeat(POLICY_DIM, axis=1), np.zeros(
            (features1.shape[0], 1), dtype=np.float32
        )

    adapter._run_iobinding = fake_run_iobinding

    n = 4
    f1 = np.zeros((n, FEATURES1_NUM, 9, 9), dtype=np.float32)
    f2 = np.zeros((n, FEATURES2_NUM, 9, 9), dtype=np.float32)
    policy, value = adapter.run(f1, f2)

    assert seen_shapes["f1"] == (4, FEATURES1_NUM, 9, 9)
    assert policy.shape == (n, POLICY_DIM)
    assert value.shape == (n, 1)


def test_run_with_no_fixed_batch_size_passes_arrays_through_unpadded():
    adapter = _make_unconstructed_adapter(fixed_batch_size=None)
    seen_shapes = {}

    def fake_run_iobinding(features1, features2):
        seen_shapes["f1"] = features1.shape
        return (
            np.zeros((features1.shape[0], POLICY_DIM), dtype=np.float32),
            np.zeros((features1.shape[0], 1), dtype=np.float32),
        )

    adapter._run_iobinding = fake_run_iobinding

    n = 3
    f1 = np.zeros((n, FEATURES1_NUM, 9, 9), dtype=np.float32)
    f2 = np.zeros((n, FEATURES2_NUM, 9, 9), dtype=np.float32)
    policy, value = adapter.run(f1, f2)

    # No fixed batch size configured: the underlying session sees exactly
    # what the caller passed in, no padding at all.
    assert seen_shapes["f1"] == (n, FEATURES1_NUM, 9, 9)
    assert policy.shape == (n, POLICY_DIM)
    assert value.shape == (n, 1)


# ---------------------------------------------------------------------------
# Real onnxruntime session construction, provider resolution, and io_binding.
# ---------------------------------------------------------------------------


def test_adapter_loads_and_runs_a_real_session(tiny_book_model_path):
    adapter = OnnxRuntimeInferenceSession(tiny_book_model_path, device_id=0)

    # A provider was resolved, and it is one onnxruntime actually reports
    # as available in this process (see this module's docstring: on a
    # GPU-less machine this is expected to be CPUExecutionProvider).
    assert adapter.resolved_provider is not None
    assert adapter.resolved_provider in adapter._session.get_providers()

    n = 5
    rng = np.random.default_rng(42)
    f1 = rng.standard_normal((n, FEATURES1_NUM, 9, 9)).astype(np.float32)
    f2 = rng.standard_normal((n, FEATURES2_NUM, 9, 9)).astype(np.float32)

    policy, value = adapter.run(f1, f2)

    assert policy.shape == (n, POLICY_DIM)
    assert value.shape[0] == n
    assert np.all(np.isfinite(policy))
    assert np.all(np.isfinite(value))
    # The model's value head ends in a Sigmoid, so every output_value must
    # lie in [0, 1] regardless of the (zero-weight) matmul beneath it --
    # Requirement 5.5's win-rate range, at the model-contract level.
    assert np.all((value >= 0.0) & (value <= 1.0))


def test_adapter_pads_a_short_batch_against_a_real_session(tiny_book_model_path):
    """End-to-end: a fixed_batch_size larger than the call's n rows still
    returns correctly-shaped, finite output for exactly n rows, through a
    real onnxruntime session (not a stub), closing the gap the pure-logic
    tests above leave between "the padding array shapes are right" and
    "a real session accepts and returns those padded shapes correctly".
    """
    adapter = OnnxRuntimeInferenceSession(
        tiny_book_model_path, device_id=0, fixed_batch_size=8
    )

    n = 3
    rng = np.random.default_rng(7)
    f1 = rng.standard_normal((n, FEATURES1_NUM, 9, 9)).astype(np.float32)
    f2 = rng.standard_normal((n, FEATURES2_NUM, 9, 9)).astype(np.float32)

    policy, value = adapter.run(f1, f2)

    assert policy.shape == (n, POLICY_DIM)
    assert value.shape[0] == n
    assert np.all(np.isfinite(policy))
    assert np.all(np.isfinite(value))


def test_adapter_falls_back_gracefully_when_gpu_providers_are_unavailable(tiny_book_model_path, caplog):
    """On this GPU-less machine, requesting Tensorrt/CUDA must not raise;
    onnxruntime's own fallback plus this adapter's CPUExecutionProvider
    tail must produce a working session (see module docstring: the actual
    TensorRT-vs-CUDA distinction cannot be exercised without GPU hardware,
    so this test only asserts the fallback does not crash and lands on a
    provider onnxruntime actually reports).
    """
    import logging

    with caplog.at_level(logging.INFO, logger="dlshogi.book.evaluator"):
        adapter = OnnxRuntimeInferenceSession(tiny_book_model_path, device_id=0)

    assert adapter.resolved_provider in ort_available_providers()
    assert any("resolved provider" in record.message for record in caplog.records)


def ort_available_providers():
    import onnxruntime as ort

    return ort.get_available_providers()
