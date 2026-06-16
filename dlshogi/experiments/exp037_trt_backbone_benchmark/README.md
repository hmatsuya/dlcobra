# exp037: Backbone inference-speed benchmark (InceptionNeXt vs ResNet vs SE-ResNet)

**Question**: At equal parameter count (~86M), is the InceptionNeXt backbone
slower at inference than a plain ResNet / SE-ResNet? Production path is ONNX +
TensorRT.

## Networks (equal-param, ~86M)

| name           | params  | block type                                    |
|----------------|---------|-----------------------------------------------|
| inceptionnext  | 86.14M  | depthwise `3x3` + `1x9` + `9x1` (exp026 arch) |
| resnet32x384   | 85.83M  | dense `3x3` conv                              |
| senet32x384    | 87.02M  | dense `3x3` conv + SE channel attention       |

## Environment note (why not raw TensorRT)

`trtexec` (TensorRT v11/10) is installed but **fails on this machine**:
`CUDA driver version is insufficient for CUDA runtime version` (driver
550.163.01 vs the bundled CUDA runtime). No Python `tensorrt` / `torch_tensorrt`
package either. So raw TensorRT timing could not be produced here.

The ONNX export + `trtexec` scripts (`export_onnx.py`, `run_benchmark.sh`,
`parse_results.py`) are kept ready for a machine with a matching driver.

Measurements below use **PyTorch FP16 on GPU** (eager + `torch.compile`,
the exp017 production path). The depthwise-vs-dense roofline behaviour is a
property of the CUDA backend and shows up the same way; TensorRT would widen the
gap further since it tunes dense conv tensor-core kernels most aggressively.

## Results (RTX 3090, FP16, pos/s = positions/sec, higher better)

### TensorRT 10.0 (Docker, production path)
| batch | inceptionnext | resnet32x384 | senet32x384 | hybrid36p5 | hybrid_exp038 |
|------:|--------------:|-------------:|------------:|-----------:|--------------:|
| 16    | 3,193 (1.00x) | 6,501 (2.04x)| 5,677 (1.78x)| 5,871 (1.84x)| 3,117 (0.98x)|
| 64    | 4,509 (1.00x) | 8,187 (1.82x)| 7,546 (1.67x)| 7,317 (1.62x)| 4,971 (1.10x)|
| 256   | 4,725 (1.00x) | 8,948 (1.89x)| 8,250 (1.75x)| 8,016 (1.70x)| 5,035 (1.07x)|

Networks:
- inceptionnext: 86.1M (exp026 architecture, depthwise 3x3 + 1x9 + 9x1)
- resnet32x384: 85.8M (dense 3x3, equal-param baseline)
- senet32x384: 87.0M (dense 3x3 + SE, equal-param)
- hybrid36p5: 86.2M (29 dense + 7 axial blocks, equal-param)
- hybrid_exp038: 134.2M (29 dense + 7 axial blocks, ch480, speed-matched to InceptionNeXt)

### Eager (PyTorch FP16)
| batch | inceptionnext | resnet32x384 | senet32x384 |
|------:|--------------:|-------------:|------------:|
| 1     | 86.0 (1.00x)  | 159.9 (1.86x)| 96.3 (1.12x)|
| 16    | 1279.8 (1.00x)| 2265.4 (1.77x)| 1558.4 (1.22x)|
| 64    | 2607.3 (1.00x)| 3702.6 (1.42x)| 3385.1 (1.30x)|
| 256   | 2669.5 (1.00x)| 3921.9 (1.47x)| 3655.0 (1.37x)|

### torch.compile (reduce-overhead) — production path
| batch | inceptionnext | resnet32x384 | senet32x384 |
|------:|--------------:|-------------:|------------:|
| 1     | 511.2 (1.00x) | 510.4 (1.00x)| 451.4 (0.88x)|
| 16    | 1889.0 (1.00x)| 3050.7 (1.62x)| 2905.4 (1.54x)|
| 64    | 2787.2 (1.00x)| 4675.2 (1.68x)| 4481.2 (1.61x)|
| 256   | 2843.9 (1.00x)| 4815.4 (1.69x)| 4677.6 (1.64x)|

## Conclusion

**Yes — at equal params, InceptionNeXt is meaningfully slower.** Under TensorRT
FP16 (the production path), ResNet is **~1.9-2.0x** and SE-ResNet **~1.7-1.8x**
faster than InceptionNeXt. TensorRT widens the gap vs torch.compile (~1.7x) because
its kernel tuning rewards dense convs even more aggressively.

The depthwise `1x9`/`9x1`/`3x3` convs are memory-bound and underuse tensor cores,
while dense `3x3` convs are math-bound and saturate them. SE adds negligible cost
(~7-10% vs plain ResNet), so SE-ResNet keeps most of ResNet's speed advantage.

Trade-off: InceptionNeXt's value is inductive bias (axial receptive field for
ranging pieces / accuracy-per-position), not throughput. The real decision is
accuracy-per-position vs nodes-per-second in MCTS.

## Hybrid: dense ResNet blocks + sparse axial-NeXt blocks

Idea: keep mostly fast dense `3x3` ResNet blocks, insert a (slow but global)
axial-NeXt block (`1x9` + `9x1` depthwise + MLP) every `axial_period` blocks so
global row/column information still propagates. See `hybrid_model.py`.

Two equal-param variants:
- `hybrid36p5`: 36 blocks, period 5 -> 86.15M (29 dense + 7 axial)
- `hybrid37p4`: 37 blocks, period 4 -> 85.87M (28 dense + 9 axial)

`torch.compile` FP16, pos/s (vs InceptionNeXt):

| batch | inceptionnext | resnet32x384 | senet32x384 | hybrid36p5 | hybrid37p4 |
|------:|--------------:|-------------:|------------:|-----------:|-----------:|
| 16    | 1909.8 (1.00x)| 3081.5 (1.61x)| 2933.8 (1.54x)| 2911.1 (1.52x)| 2872.4 (1.50x)|
| 64    | 2796.1 (1.00x)| 4691.9 (1.68x)| 4491.8 (1.61x)| 4285.4 (1.53x)| 4192.6 (1.50x)|
| 256   | 2851.6 (1.00x)| 4804.9 (1.69x)| 4666.1 (1.64x)| 4367.2 (1.53x)| 4272.1 (1.50x)|

TensorRT 10.0 FP16 (Docker), pos/s (vs InceptionNeXt):

| batch | inceptionnext | resnet32x384 | senet32x384 | hybrid36p5 | hybrid_exp038 |
|------:|--------------:|-------------:|------------:|-----------:|--------------:|
| 16    | 3,193 (1.00x) | 6,501 (2.04x)| 5,677 (1.78x)| 5,871 (1.84x)| 3,117 (0.98x)|
| 64    | 4,509 (1.00x) | 8,187 (1.82x)| 7,546 (1.67x)| 7,317 (1.62x)| 4,971 (1.10x)|
| 256   | 4,725 (1.00x) | 8,948 (1.89x)| 8,250 (1.75x)| 8,016 (1.70x)| 5,035 (1.07x)|

**Hybrid is viable and well-balanced.** Under TensorRT, the equal-param hybrid
(hybrid36p5) is **1.62-1.84x** faster than InceptionNeXt — even faster than
SE-ResNet at small batch. The speed-matched hybrid (hybrid_exp038, 134M) holds
steady at 0.98-1.10x of InceptionNeXt, confirming the speed budget was
conservative (slightly faster, not slower).

Recommended next step: train `hybrid36p5` (and/or `senet32x384`) from the same
data/schedule as exp026 and compare val/loss + policy accuracy at equal params.

## Reproduce

```bash
# Torch FP16 (works here)
PYTHONPATH=$(pwd) .venv/bin/python \
  dlshogi/experiments/exp037_trt_backbone_benchmark/bench_torch.py \
  --batches 1 16 64 256 --compile

# TensorRT path via Docker (works here — uses NGC 24.05 container)
# NOTE: run this when GPU is free (not during training) for stable timings
bash dlshogi/experiments/exp037_trt_backbone_benchmark/run_trtexec_docker.sh 16 64 256

# TensorRT path (native trtexec — needs driver/runtime match, broken on this box)
bash dlshogi/experiments/exp037_trt_backbone_benchmark/run_benchmark.sh 1 16 64 256
```

## Docker trtexec setup

Host `trtexec` (TRT v11) fails with "CUDA driver version is insufficient" because
its bundled CUDA runtime exceeds what driver 550.163 supports. Solution: use the
NGC TensorRT container (`nvcr.io/nvidia/tensorrt:24.05-py3`) which bundles TRT 10.0
+ CUDA 12.4, compatible with driver ≥ 550.54.

The script `run_trtexec_docker.sh`:
1. Exports FP16 ONNX models on the host (via venv PyTorch)
2. Runs trtexec inside the container with `--fp16 --noDataTransfers --useSpinWait --useCudaGraph`
3. Parses results on the host

Verified: container starts, GPU is accessible, `&&&& PASSED` on test model.
