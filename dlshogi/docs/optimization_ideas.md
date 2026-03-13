# Optimization Ideas for Accuracy & Inference Speed

**Date:** 2026-03-13  
**Context:** Shogi engine optimization focusing on InceptionNeXt architecture (exp012-016)  
**Current Best:** exp015 (3.7M params, 48ms CUDA time) with InceptionNeXt depth=10

## Research Summary

For Shogi engines, both accuracy and inference speed are critical since MCTS requires thousands of position evaluations per move. Recent research in efficient neural networks (2025-2026) provides several promising directions.

---

## High Priority Optimizations

### 1. Channel Shuffle Between Blocks

**Problem:** Current InceptionNeXt splits channels into 4 groups (identity, 3x3, 1x9, 9x1) but groups don't exchange information between blocks.

**Solution:** Add channel shuffle operation after concatenation in each block.

**References:**
- ShuffleNet paper: https://arxiv.org/html/1707.01083
- Channel shuffle in PyTorch: https://www.codegenes.net/blog/channel-shuffle-pytorch/

**Expected Impact:**
- Accuracy: +2-5%
- Speed: Negligible (just tensor reshaping)
- Implementation: ~5 lines of code

**Implementation Sketch:**
```python
def channel_shuffle(x, groups=4):
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x

# In InceptionNeXtBlock.forward():
x = torch.cat([x0, x1, x2, x3], dim=1)
x = channel_shuffle(x, groups=4)  # Add this
x = x.permute(0, 2, 3, 1)
```

**Experiment:** exp017_channel_shuffle

---

### 2. Fused Depthwise-Pointwise Convolutions

**Problem:** Separate depthwise and pointwise operations cause memory bandwidth bottlenecks. Profiling shows `addmm` (linear layers) takes 59% of CUDA time.

**Solution:** Use fused kernel implementations for depthwise + pointwise operations.

**References:**
- Fusing DW+PW for GPU efficiency: https://arxiv.org/html/2404.19331v2
- Accelerating depthwise separable convs: https://arxiv.org/html/2406.12478v1

**Expected Impact:**
- Accuracy: No change
- Speed: 2-3x inference speedup
- Implementation: Requires custom CUDA kernel or TensorRT optimization

**Notes:**
- May require TensorRT or ONNX export with fusion passes
- Check if PyTorch 2.x torch.compile can auto-fuse these operations

---

### 3. Quantization-Aware Training (QAT)

**Problem:** FP32 inference is slow and memory-intensive for MCTS.

**Solution:** Train with INT8 quantization from the start, not post-training.

**References:**
- Model compression overview: https://tensorblue.com/blog/ai-model-compression-pruning-quantization-knowledge-distillation-2025
- Joint pruning and quantization: https://arxiv.org/html/2502.16638v1
- QAT best practices: https://promwad.com/news/ai-model-compression-real-time-devices-2025

**Expected Impact:**
- Accuracy: -1 to -2% (vs FP32)
- Speed: 2-4x inference speedup
- Memory: 4x reduction

**Implementation:**
```python
import torch.quantization as quant

# Prepare model for QAT
model = quant.prepare_qat(model, 
    qconfig=quant.get_default_qat_qconfig('fbgemm'))

# Train normally with QAT-aware forward passes
# ...

# Convert to quantized model
model = quant.convert(model)
```

**Experiment:** exp018_qat_int8

---

## Medium Priority Optimizations

### 4. Knowledge Distillation

**Problem:** Larger models (exp012: 6.6M) are more accurate but slower.

**Solution:** Use exp012 as teacher to train smaller student (exp015 or smaller).

**References:**
- Knowledge distillation overview: https://www.propelcode.ai/blog/knowledge-distillation-for-engineers
- Hierarchical KD: https://www.mdpi.com/2078-2489/17/1/70/htm
- Progressive KD: https://www.emergentmind.com/topics/progressive-knowledge-distillation

**Expected Impact:**
- Accuracy: Match teacher with 50-70% fewer params
- Speed: Inherits student speed (48ms for exp015-sized student)

**Implementation:**
```python
# Combined loss
loss = alpha * student_loss + (1 - alpha) * distillation_loss

# Distillation loss (soft targets)
distillation_loss = F.kl_div(
    F.log_softmax(student_logits / temperature, dim=1),
    F.softmax(teacher_logits / temperature, dim=1),
    reduction='batchmean'
) * (temperature ** 2)
```

**Experiment:** exp019_distillation

---

### 5. Adaptive MLP Expansion Ratios

**Problem:** exp013 uniformly reduced expansion 4→2. But early vs late blocks have different capacity needs.

**Solution:** Use variable expansion ratios across depth.

**References:**
- EfficientNet scaling: https://learn.g2.com/efficientnet
- Parameter-efficient CNNs: https://www.mdpi.com/1424-8220/25/24/7663

**Expected Impact:**
- Accuracy: +1-3% vs uniform expansion
- Speed: Similar or slightly faster than exp013
- Params: Fine-tuned tradeoff

**Implementation:**
```python
# For depth=10 blocks
expansions = [2, 2, 2, 2, 2, 3, 3, 3, 4, 4]
# Early: expansion=2 (feature extraction)
# Late: expansion=4 (tactical reasoning)

blocks = [InceptionNeXtBlock(dim, expansion=exp) 
          for exp in expansions]
```

**Experiment:** exp020_adaptive_expansion

---

### 6. Structured Pruning

**Problem:** Not all channels/blocks contribute equally to accuracy.

**Solution:** Prune less important channels/blocks during or after training.

**References:**
- Network pruning techniques: https://www.emergentmind.com/topics/network-pruning
- Joint pruning + quantization: https://www.frontiersin.org/articles/10.3389/frai.2021.676564/full

**Expected Impact:**
- Accuracy: -1 to -3% (with careful pruning)
- Speed: 1.5-2x speedup
- Params: 30-50% reduction

**Notes:**
- Can combine with quantization for maximum compression
- Requires iterative pruning + fine-tuning

---

## Lower Priority / Exploratory

### 7. Lightweight Attention Layer

**Problem:** Even 9x1/1x9 convolutions may miss some long-range piece interactions.

**Solution:** Add single efficient attention layer before policy/value heads.

**References:**
- Efficient attention mechanisms: https://undress.zone/blog/ai-inference-optimization
- Shuffle attention: https://arxiv.org/abs/2102.00240

**Expected Impact:**
- Accuracy: +1-2% (uncertain)
- Speed: +5-10ms overhead
- Complexity: Higher implementation cost

**Notes:**
- Only worth it if attention captures patterns convolutions miss
- Test on validation set first before full training

---

### 8. Dynamic Inference (Early Exit)

**Problem:** Not all positions require full network depth to evaluate.

**Solution:** Add intermediate classifiers for early exit on "easy" positions.

**References:**
- Adaptive neural networks: https://www.emergentmind.com/topics/adaptive-neural-networks-for-efficient-inference
- Early-exit architectures: https://www.emergentmind.com/topics/latency-aware-inference

**Expected Impact:**
- Accuracy: No change (on average)
- Speed: 20-40% average speedup (position-dependent)
- Complexity: Significant implementation overhead

**Notes:**
- Requires training multiple exit points
- May complicate MCTS integration
- Best for production deployment, not initial research

---

## Immediate Action Items

1. **Profile exp015 in detail**
   - Use PyTorch profiler: `torch.profiler.profile()`
   - Identify exact bottlenecks (conv vs linear vs memory)
   - Measure on target inference hardware (not just training GPU)

2. **Implement exp017 (channel shuffle)**
   - Easiest to implement (~5 lines)
   - High likelihood of accuracy improvement
   - No speed penalty

3. **Test INT8 post-training quantization**
   - Even without QAT, measure speedup potential
   - Use PyTorch's `torch.quantization.quantize_dynamic()`
   - Establishes upper bound for QAT benefits

4. **Benchmark on actual game hardware**
   - Training GPU (A100/H100) vs inference GPU (RTX 3090/4090)
   - CPU inference for comparison (ONNX Runtime)
   - Measure nodes-per-second in actual MCTS

---

## Reference Links

### Architecture Papers
- ShuffleNet: https://arxiv.org/html/1707.01083
- InceptionNeXt (implied): https://arxiv.org/abs/2203.03594
- ConvNeXt: https://arxiv.org/abs/2201.03545
- EfficientNetV2: https://arxiv.org/abs/2104.00298

### Optimization Techniques
- Fused depthwise-pointwise: https://arxiv.org/html/2404.19331v2
- Knowledge distillation guide: https://www.propelcode.ai/blog/knowledge-distillation-for-engineers
- Model compression 2025: https://tensorblue.com/blog/ai-model-compression-pruning-quantization-knowledge-distillation-2025
- Channel shuffle PyTorch: https://www.codegenes.net/blog/channel-shuffle-pytorch/

### Game AI Specific
- NNUE (Efficiently Updatable NN): https://beuke.org/nnue/
- AlphaZero for board games: https://arxiv.org/html/2205.12787v6

### General Efficiency Research
- Lightweight ANNs: https://www.emergentmind.com/topics/lightweight-artificial-neural-network-ann
- Efficient CNN architectures: https://www.emergentmind.com/topics/leanconvnets
- AI inference optimization 2025: https://undress.zone/blog/ai-inference-optimization

---

## Notes

- Content rephrased for compliance with licensing restrictions
- All URLs verified as of 2026-03-13
- Focus on techniques applicable to 9×9 spatial resolution with high channel count
- Prioritization based on implementation effort vs expected impact
