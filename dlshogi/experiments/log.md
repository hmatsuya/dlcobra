# Experiment Log

### exp006: Single cycle cosine annealing
**日付**: 2026-03-05
**ベース実験**: exp004
**改善内容**:
- 単一サイクルのcosine annealingに変更: cycle_limit=8→1
- サイクル長を大幅に延長: t_initial=300000→3000000
- warmup期間を調整: warmup_t=30000→150000（t_initialの5%）
- cycle_mul=1.0（サイクル長を変えない）、cycle_decay=1.0（減衰なし）
- 長期間の単一サイクルで学習率を緩やかに減衰させる効果を検証

### exp005: Extended warmup with higher initial LR
**日付**: 2026-03-04
**ベース実験**: exp004
**改善内容**:
- warmup開始学習率をさらに引き上げ: warmup_lr_init=1e-6→1e-5
- exp004との比較により、warmup初期学習率の影響を検証
- より高い初期学習率での収束性と安定性を評価

### exp004: Extended warmup only (no EMA)
**日付**: 2026-03-04
**ベース実験**: なし（ベース設定からの変更）
**改善内容**:
- warmup期間を大幅に延長: warmup_t=10→30000（t_initialの10%）
- warmup開始学習率を調整: warmup_lr_init=1e-7→1e-6
- EMAは使用せず、warmup単独の効果を検証
- exp003との比較により、EMAの追加効果を測定可能

### exp003: Extended warmup and EMA testing
**日付**: 2026-03-04
**ベース実験**: なし（ベース設定からの変更）
**改善内容**:
- EMAを有効化（use_ema=true, update_bn=true）
- EMA設定: ema_start_epoch=0（最初から開始）, ema_freq=100, ema_decay=0.99
- warmup期間を大幅に延長: warmup_t=10→30000（t_initialの10%）
- warmup開始学習率を調整: warmup_lr_init=1e-7→1e-6
- EMAによるモデル重みの平滑化と、長いwarmupによる安定した学習開始の効果を検証

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
