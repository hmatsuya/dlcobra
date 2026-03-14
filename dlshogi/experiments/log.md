# Experiment Log

### exp021: InceptionNeXt + Squeezeformer tail block
**日付**: 2026-03-14
**ベース実験**: exp015
**パラメータ数**: 4.3M
**改善内容**:
- exp015の最後のInceptionNeXtブロックをSqueezeformerブロック（NeurIPS 2022）に置換
- 構成: 9 InceptionNeXtブロック + 1 Squeezeformerブロック
- Squeezeformerブロック: MHA+LN → FFN+LN → ConvModule+LN → FFN+LN（全て残差接続付き）
- 9×9盤面をseq_len=81に平坦化し、相対位置エンコーディング付きMHAで大域的な駒の関係を捉える
- 局所特徴抽出（InceptionNeXt）→ 大域的注意機構（Squeezeformer）のハイブリッド設計
- 初回実行でval/loss=nanが発生。原因: 16-mixed (FP16)でattentionスコアがオーバーフロー（FP16の最大値65504を超過）
- 対策: precision を bf16-mixed に変更（BF16はFP32と同じ指数部8bitで動的範囲が広い）

### exp020: Adaptive MLP Expansion Ratios
**日付**: 2026-03-14
**ベース実験**: exp015
**パラメータ数**: 3.4M
**改善内容**:
- MLP expansion ratioをブロック深さに応じて可変化（3:6:3配分、12ブロック）
- expansions=[2,2,2,3,3,3,3,3,3,4,4,4]
- 早期(1-3): expansion=2、核心(4-9): expansion=3、後期(10-12): expansion=4
- EfficientNetのcompound scalingに着想、深さに応じた容量配分で効率改善を狙う
- 推論速度: 8.0ms（batch=128）、exp015比でブロック数1.2倍、パラメータ数は8%削減(3.4M vs 3.7M)

### exp019: InceptionNeXt with 5x5 Depthwise Branch
**日付**: 2026-03-14
**ベース実験**: exp015
**パラメータ数**: 3.6M
**改善内容**:
- InceptionNeXtブロックを4並列→5並列に拡張（5x5 depthwise convを追加）
- チャンネル分割: dim // 5 each for identity, 3x3, 1x9, 9x1, 5x5
- 5x5ブランチが桂馬のL字ジャンプ・角の斜め移動を1層で捉える
- dims=[190]（5で割り切れる必要があるため192→190に変更）
- depthwiseのためFLOP増加は微小、受容野は大幅に拡大
- 推論速度: 9.5ms（exp015: 7.8ms、1.22x遅い、batch=128）

### exp018: Lighter Value Head (2 channels)
**日付**: 2026-03-14
**ベース実験**: exp015
**パラメータ数**: 3.1M
**改善内容**:
- Value Headのボトルネック修正: MAX_MOVE_LABEL_NUM (27) → 2チャネルに削減
- value_fc1の入力次元: 2187 → 162 (約13倍削減)
- パラメータ削減: 523k (14.3%削減、3.67M→3.14M)
- 推論速度: 7.55ms (exp015: 7.61ms、1.01x高速化、batch=128)
- AlphaZero/KataGoと同様の設計パターン

### exp017: PyTorch Compiler & Memory Layout Optimization
**日付**: 2026-03-14
**ベース実験**: exp015
**パラメータ数**: 3.7M
**改善内容**:
- torch.compileとchannels_lastメモリフォーマットによる推論高速化を検証
- 精度は変わらず、推論速度のみ改善（プロファイリング専用実験）
- 結果: batch=128で1.30x、batch=1024で1.22x高速化
- channels_lastは追加効果が小さい（~0.01x）ため、torch.compile単独で十分
- 本番デプロイ時はONNX→TensorRT FP16で更に2-3x高速化が期待できる

### exp016: Focal Lossへの変更
**日付**: 2026-03-13
**ベース実験**: exp015
**パラメータ数**: 3.7M
**改善内容**:
- ポリシーヘッドをCross Entropy→Focal Loss: FL(pt) = -α(1-pt)^γ * log(pt)
- バリューヘッド（result/value両方）をBCE→Binary Focal Loss: BFL = α(1-pt)^γ * BCE
- パラメータ: focal_alpha=0.25、focal_gamma=2.0（RetinaNet標準値）
- `ptl.py`は無変更、`FocalModel`サブクラスを`ptl_focal.py`に実装（実験ディレクトリ内で完結）
- ネットワーク構成はexp015と同一（InceptionNeXt depths=[10], dims=[192]）、CUDA時間: 48ms

### exp015: Isotropic InceptionNeXt、深さ半減
**日付**: 2026-03-12
**ベース実験**: exp012
**パラメータ数**: 3.7M
**改善内容**:
- exp012（depths=[20], dims=[192]）からdepthsを半減: depths=[10], dims=[192]
- ブロック構成・MLP expansion=4はexp012と同一
- パラメータ数はexp012の6.6M→3.7M（約44%削減）
- 深さと精度のトレードオフを検証

### exp014: 階層型InceptionNeXt（Hierarchical）
**日付**: 2026-03-12
**ベース実験**: exp013
**パラメータ数**: 3.1M
**改善内容**:
- 等幅（isotropic）設計から階層型（depths=[3,3,9,3], dims=[64,128,192,256]）に変更
- ステージ間をLayerNorm+Linearでチャネル数を段階的に拡大（チャネル射影）
- MLP expansion=2（exp013と同様）を採用
- exp012比でパラメータ数約50%削減（6.1M→3.1M）、CUDA時間も27%削減（94ms→69ms）
- depthwiseコンボリューションのコストは固定のまま、MLP FLOPsを大幅削減

### exp013: InceptionNeXt expansion=2
**日付**: 2026-03-11
**ベース実験**: exp012
**パラメータ数**: 3.7M
**改善内容**:
- MLPのexpansion ratioを4→2に削減（192→384→192、exp012は192→768→192）
- プロファイリングでaddmm（線形層）がCUDA時間の59%を占めることを確認し、その削減が目的
- パラメータ数はexp012の6.6M→3.7Mに減少（約44%削減）
- 速度とモデル容量のトレードオフを検証

### exp012: Isotropic InceptionNeXt
**日付**: 2026-03-11
**ベース実験**: exp011
**パラメータ数**: 6.6M
**改善内容**:
- ConvNeXtのdepthwise 3x3をInceptionNeXt式の4並列ブランチに変更
- チャンネルを4等分し、identity / 3x3 / 1x9 / 9x1 の各depthwise convを並列適用
- 1x9・9x1カーネルが将棋の飛車・香車の直線移動を1層で捉えるinductive biasを持つ
- depths=[20], dims=[192]はexp011と同一構成で直接比較可能

### exp011: ConvNeXt flat deep (single-stage)
**日付**: 2026-03-10
**ベース実験**: exp010
**パラメータ数**: 6.7M
**改善内容**:
- exp010（depths=[10], dims=[256]）からdepths=[20], dims=[192]に変更
- より深く・より細いネットワークとの比較実験

### exp010: ConvNeXt flat (single-stage)
**日付**: 2026-03-10
**ベース実験**: exp008
**パラメータ数**: 6.0M
**改善内容**:
- 4ステージ構成からフラット（単一ステージ）に変更: depths=[10], dims=[256]
- チャンネル数を全ブロックで均一（256ch）に統一、ステージ間のダウンサンプリングなし
- ConvNeXtブロックはexp008と同じ（depthwise conv + LayerNorm + inverted bottleneck + GELU）
- 基本のresnet10（192ch）との比較: ConvNeXtブロック × フラット構成

### exp009: ResNet-style channel distribution
**日付**: 2026-03-10
**ベース実験**: exp008
**パラメータ数**: 20.5M
**改善内容**:
- exp008と同じチャンネル分布を維持: depths=[1, 1, 3, 1], dims=[96, 192, 384, 768]
- ConvNeXtブロック（depthwise conv + LayerNorm + inverted bottleneck）を通常のResNetブロックに変更
- ResNetブロック: BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv3x3 + residual
- 正規化をLayerNormからBatchNormに変更、活性化関数をGELUからReLUに変更
- ConvNeXt vs ResNetのアーキテクチャ比較実験

### exp008: ConvNeXt-style channel distribution
**日付**: 2026-03-10
**ベース実験**: exp001
**改善内容**:
- ConvNeXtのチャンネル分布を採用: depths=[1, 1, 3, 1], dims=[96, 192, 384, 768]
- 4ステージ構成でチャンネル数を段階的に増加（96→192→384→768）
- 各ブロックはdepthwise conv + LayerNorm + inverted bottleneck (×4) + GELUのConvNeXt構造
- ステージ間はLayerNorm + 1x1 convでチャンネル数を変換（9x9盤面のため空間解像度は維持）
- 活性化関数はGELU、正規化はLayerNorm（BatchNormの代わり）

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
