# Experiment Log

### exp001: fewer activations

**日付**: 2026-03-03

**ベース実験**: なし（新規）

**改善内容**:
- ResNetブロック内の活性化関数を2つから1つに削減（ConvNeXtスタイル）
- 標準: Conv -> BN -> Act -> Conv -> BN -> (+residual) -> Act
- 変更後: Conv -> BN -> Conv -> BN -> (+residual) -> Act
- 計算コスト削減と表現力のトレードオフを検証
- Swish (SiLU) 活性化関数を使用
