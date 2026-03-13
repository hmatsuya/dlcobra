
# Optimization Ideas for Accuracy & Inference Speed (Refined by Gemini 3.1 Pro Preview)

**Date:** 2026-03-13  
**Context:** Shogi engine optimization focusing on InceptionNeXt architecture (exp012-016)  
**Current Best:** exp016 (Focal Loss) / exp015 (3.7M params, 48ms CUDA time) with Isotropic InceptionNeXt depth=10

## Research Summary & Code Review

Recent review of the `model.py` implementation revealed major opportunities for parameter reduction in the Value Head and memory-layout optimizations. Furthermore, while the current `[id, 3x3, 1x9, 9x1]` branches beautifully capture Rooks and Lances, they lack direct receptive fields for Knights (L-shape) and Bishops (diagonals). 

---

## High Priority Optimizations

### 1. Value Head Bottleneck Fix (Parameter Reduction)

**Problem:** The Value Head uses `MAX_MOVE_LABEL_NUM` (e.g., 27+ for Shogi directions/drops) as the channel dimension before flattening. This creates a massive, unnecessarily dense Linear layer (`9 * 9 * 27 = 2187 -> fcl`). This wastes ~500,000 parameters (~14% of the exp015 model) and slows down `addmm` operations.
**Solution:** Project the Value Head to only 2 or 4 channels before flattening, as done in AlphaZero/KataGo.

**Expected Impact:**
- Accuracy: Neutral (or slight gain due to reduced overfitting)
- Speed: Noticeable reduction in linear layer CUDA time
- Params: ~500k reduction

**Implementation Sketch (in `PolicyValueNetwork.__init__`):**
```python
VALUE_CHANNELS = 2  # Reduced from MAX_MOVE_LABEL_NUM

self.value_conv = nn.Conv2d(dim, VALUE_CHANNELS, kernel_size=1, bias=False)
self.value_norm = nn.LayerNorm(VALUE_CHANNELS)
self.value_fc1 = nn.Linear(9 * 9 * VALUE_CHANNELS, fcl) # 162 -> fcl
```
**Experiment:** exp017_value_head_fix

---

### 2. The 5x5 "Knight/Bishop" Depthwise Branch

**Problem:** The current 3x3 branch requires multiple layers to propagate threats from Knight jumps (L-shape) and Bishop moves (diagonal). 
**Solution:** Expand the InceptionNeXt block to 5 parallel branches by adding a `5x5` depthwise convolution. Because it is depthwise, the FLOP increase is negligible, but it drastically improves the receptive field for Shogi's unique piece movements.

**Expected Impact:**
- Accuracy: High (+ tactical sequence evaluation)
- Speed: Negligible overhead
- Implementation: Requires adjusting `dim` to be divisible by 5 (e.g., `dim=190` or `200`).

**Implementation Sketch:**
```python
# Channel split into 5 branches (dim // 5)
self.dw3x3 = nn.Conv2d(branch_dim, branch_dim, 3, padding=1, groups=branch_dim)
self.dw1x9 = nn.Conv2d(branch_dim, branch_dim, (1, 9), padding=(0, 4), groups=branch_dim)
self.dw9x1 = nn.Conv2d(branch_dim, branch_dim, (9, 1), padding=(4, 0), groups=branch_dim)
self.dw5x5 = nn.Conv2d(branch_dim, branch_dim, 5, padding=2, groups=branch_dim) # NEW
# Forward pass: concatenate 5 branches instead of 4
```
**Experiment:** exp018_inception_5branch

---

### 3. PyTorch Compiler & Memory Layout Optimization

**Problem:** The `permute` operations (`NCHW -> NHWC -> NCHW`) inside every single block cause memory fragmentation.
**Solution:** Utilize `torch.channels_last` for native NHWC processing and `torch.compile` to fuse memory layouts and dense linear layers during training/evaluation. For actual engine deployment, export to TensorRT (FP16).

**Expected Impact:**
- Speed: Immediate 2-3x inference speedup (especially with TensorRT FP16)
- Accuracy: Zero loss

**Implementation Sketch:**
```python
model = PolicyValueNetwork(depths=[10], dims=[190])
model = model.to(memory_format=torch.channels_last)
model = torch.compile(model, mode="reduce-overhead")
```

---

## Medium Priority Optimizations

### 4. Squeeze-and-Excitation (SE) for Drop Pieces (持ち駒)

**Problem:** Shogi uniquely allows dropping captured pieces anywhere on the board. Standard convolutions struggle to efficiently broadcast a global state (e.g., "I hold a Pawn") to all 81 spatial squares simultaneously.
**Solution:** Add a lightweight Squeeze-and-Excitation (SE) module after the concatenation in the InceptionNeXt block. Global Average Pooling condenses the 9x9 board into a 1D vector, which then modulates the channels globally.

**Expected Impact:**
- Accuracy: +2-3% on Drop-heavy tactical positions
- Speed: +1-2ms overhead

**Experiment:** exp019_se_drops

---

### 5. MCTS-Based Knowledge Distillation

**Problem:** Standard KD just forces the student to mimic the teacher's raw logits. For board games, this is suboptimal.
**Solution:** Use the large exp012 model to play games (or evaluate random positions) using *shallow MCTS* (e.g., 200 playouts). Train the smaller exp015 student on the **MCTS visit counts** (Policy) and **MCTS root values** (Value). This transfers search-amplified knowledge.

**Expected Impact:**
- Accuracy: Student surpasses standard KD by mimicking search, not just heuristics.
- Implementation Cost: Requires modifying the data generation pipeline.

**Experiment:** exp020_mcts_distillation

---

### 6. Auxiliary Targets for Value Head

**Problem:** Predicting a single scalar (Win/Loss) provides a very sparse gradient signal for the complex Value Head.
**Solution:** Building on `exp016` (Focal Loss), add auxiliary loss heads. Predict the expected material advantage (piece score) or square ownership alongside the win/loss evaluation.

**Expected Impact:**
- Accuracy: Faster convergence, stronger positional understanding.

---

## Deprecated / Rejected Ideas

*   **Channel Shuffle Between Blocks:** *(Rejected)* Previously proposed to mix channels between the parallel branches. However, code review confirms that `self.pwconv1` is a dense `nn.Linear` layer. Because it computes a dot-product across *all* concatenated channels, it already acts as a perfect mathematical channel mixer. Adding a shuffle operation here is 100% mathematically redundant and would only slow down inference.
*   **Hierarchical Downsampling:** *(Rejected)* Tried in `exp014`. Unlike images, a 9x9 Shogi board holds exact geometric truth. Spatial pooling destroys piece coordinates. We will stick strictly to Isotropic (flat) architectures.

---

## Immediate Action Items

1. **Implement `exp017` (Value Head Fix)**
   - Change Value channel dimension from `MAX_MOVE_LABEL_NUM` to `2`.
   - Verify parameter drop and measure speedup.
2. **Export to TensorRT FP16**
   - Measure actual Nodes Per Second (NPS) using FP16 before considering INT8 Quantization.
3. **Implement `exp018` (5x5 Branch)**
   - Adjust `dim` from 192 to 190 (divisible by 5).
   - Add the 5x5 branch and evaluate accuracy gain.
