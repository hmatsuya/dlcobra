# Project Structure

```
DeepLearningShogi/
├── dlshogi/                    # Python training code
│   ├── ptl.py                  # Main training entry (LightningCLI)
│   ├── config.yaml             # Base training configuration
│   ├── common.py               # Constants (features, move labels)
│   ├── data_loader.py          # HCPE/HDF5 data loading
│   ├── lr_scheduler.py         # Custom learning rate schedulers
│   ├── serializers.py          # Model save/load utilities
│   ├── docs/                   # General documentation
│   │   ├── README.md           # Documentation index
│   │   ├── optimization_ideas.md  # Optimization strategies
│   │   └── PROFILING_GUIDE.md  # Profiling guide
│   ├── network/                # Neural network architectures
│   │   ├── policy_value_network.py        # Network factory
│   │   ├── policy_value_network_resnet.py # ResNet implementation
│   │   ├── policy_value_network_senet.py  # SENet variant
│   │   └── ...
│   ├── experiments/            # Experiment directories
│   │   ├── log.md              # Experiment log (all experiments)
│   │   └── expNNN_description/ # Each experiment is a Python package
│   │       ├── __init__.py
│   │       ├── config.yaml     # Overrides base config
│   │       ├── model.py        # Custom network (optional)
│   │       ├── run.sh          # Launch script
│   │       ├── profile.py      # Quick profiling (batch=128 default)
│   │       ├── profile_detailed.py  # Detailed profiling (optional)
│   │       └── docs/           # Experiment-specific docs (optional)
│   │           └── README.md   # Profiling results, analysis
│   └── utils/                  # Data conversion & analysis tools
│       ├── csa_to_hcpe*.py     # CSA → HCPE converters
│       ├── hcpe_to_hdf5.py     # HCPE → HDF5 converter
│       └── stat_*.py           # Dataset statistics
│
├── cppshogi/                   # C++ Shogi library (Apery-based)
│   ├── position.cpp/hpp        # Board state management
│   ├── generateMoves.cpp/hpp   # Legal move generation
│   ├── bitboard.cpp/hpp        # Bitboard operations
│   ├── cppshogi.cpp/h          # Feature extraction for NN
│   └── python_module.cpp       # Python bindings
│
├── usi/                        # USI engine (game play)
├── selfplay/                   # MCTS self-play training
├── build/                      # C++ build output
└── Pipfile                     # Python dependencies
```

## Key Patterns

### Network Architecture
- All networks implement `PolicyValueNetwork` class
- Dual inputs: `x1` (board features), `x2` (hand pieces + flags)
- Dual outputs: policy logits (2187 moves), value logit (win prob)
- Factory function in `policy_value_network.py` handles instantiation

### Experiment Organization
- Directory naming: `expNNN_short_description/` (underscore, importable)
- Each experiment overrides only changed values from base config
- Custom networks referenced via fully-qualified class path
- Template: copy `dlshogi/experiments/_template/` to create new experiments
- See `dlshogi/experiments/README.md` for step-by-step creation guide

### Experiment Logging
- 実装、動作確認が完了したら、`dlshogi/experiments/log.md`に実験ログを追加する
- 実験ログには以下を簡潔に記載（5行程度）:
  - 実験名
  - ベースとした実験名
  - パラメータ数（デバッグ実行時のモデルサマリーから取得）
  - 改善内容
- 新しいログは上に追加していく（最新が先頭）

Example:
```markdown
### exp116: Focal Lossへの変更
**日付**: 2025-12-09
**ベース実験**: exp113
**パラメータ数**: 20.5M
**改善内容**:
- BCE (Binary Cross Entropy) LossをFocal Lossに変更
- Focal Lossはクラス不均衡問題に対処する損失関数
- パラメータ: alpha=0.25、gamma=2.0(標準的な値を使用)
- 数式: FL(pt) = -α(1-pt)^γ * log(pt)
- 予測が難しいサンプル (確信度が低い) に重みを置く設計
```

### New Experiment Checklist
新しい実験を作成したら以下を実施する:
1. デバッグモードで動作確認: `bash dlshogi/experiments/expNNN_.../run.sh --debug`
2. モデルサマリーのTrainable paramsを確認し、`log.md`の`**パラメータ数**`に記載
3. プロファイリングで計算コストを確認: `bash dlshogi/experiments/expNNN_.../profile.sh` (デフォルトbatch=128)
   - 推論速度を測定する場合は、torch.compileを適用してベンチマークを実施
   - exp017の結果: 1.22-1.30x高速化（精度は変わらず）
4. (オプション) 詳細プロファイリング: `python dlshogi/experiments/expNNN_.../profile_detailed.py`
5. (オプション) 有望な実験の場合、`docs/`フォルダに分析結果を保存

### Inference Optimization (推論最適化)
推論速度を最適化する場合:
1. **torch.compile**: 1.2-1.3x高速化、精度は変わらず（exp017で検証済み）
   ```python
   model = model.eval()
   model = torch.compile(model, mode="reduce-overhead")
   ```
2. **ONNX + TensorRT**: 更に2-3x高速化が期待できる（本番デプロイ時）
   ```bash
   python dlshogi/convert_model_to_onnx.py <model_path> <output_path>
   ```
3. **合計**: ベースラインから約3-4x高速化が可能

### Data Pipeline
- Raw: CSA/KIF game records
- Intermediate: HCPE (compressed positions + evaluations)
- Training: HDF5 (batched, memory-mapped)

### Feature Dimensions
- FEATURES1_NUM: 62 channels (piece positions, attacks)
- FEATURES2_NUM: 57 channels (hand pieces, check, nyugyoku)
- Board: 9×9 grid
- Move labels: 27 directions × 81 squares = 2187
