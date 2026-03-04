# Experiment Log

### exp002: LRスケジューリングのパラメータ調整（t_initial=50000, warmup_t=5000）
**日付**: 2026-03-03
**ベース実験**: なし（ベース設定からの変更）
**改善内容**:
- CosineLRSchedulerのt_initialを300000→50000に短縮（より短いサイクル）
- warmup_tを10→5000に増加（より長いウォームアップ期間）
- 短いサイクルと長いウォームアップの組み合わせによる学習率スケジュールの効果を検証

### exp001: fewer activations

**日付**: 2026-03-03

**ベース実験**: なし（新規）

**改善内容**:
- ResNetブロック内の活性化関数を2つから1つに削減（ConvNeXtスタイル）
- 標準: Conv -> BN -> Act -> Conv -> BN -> (+residual) -> Act
- 変更後: Conv -> BN -> Conv -> BN -> (+residual) -> Act
- 計算コスト削減と表現力のトレードオフを検証
- Swish (SiLU) 活性化関数を使用
