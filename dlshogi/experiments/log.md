# Experiment Log

### exp036: exp029のresume継続（patience拡大）
**日付**: 2026-05-31
**ベース実験**: exp029
**パラメータ数**: 86.1M
**改善内容**:
- exp029のlast.ckpt（step=141,250、LR=0.000555）からLightning resumeで完全に状態を引き継ぎ
- EarlyStopping patience: 20→50に拡大（cosineスケジュールの残り約160,000ステップを走らせる）
- KD + flip augはexp029と同一設定を維持
- 動機: exp029はcosineスケジュールの47%で早期停止。LR低下に伴う改善余地を探る

**不具合と修正**:
- 初回実行はconfigのpatience=50が効かず即early stop（fit.log: "did not improve in the last 21 records"）
- 原因: LightningのEarlyStoppingはcheckpointからpatience/wait_countを復元するため、configの値が上書きされる。exp029のlast.ckptは"停止済み"状態（wait_count=20, patience=20, reason=PATIENCE_EXHAUSTED）で保存されており、resume直後の1回の非改善でwait_count=21→即停止していた
- 修正: ptl.pyは編集せず、`patch_ckpt_patience.py`でcheckpointのEarlyStopping状態のみ書き換え（wait_count=0, patience=50, reason=NOT_STOPPED）。重み・optimizer・LR scheduler・global_stepは不変。run.shはパッチ済みckpt（last_patience50.ckpt）から再開するよう更新
- debug実行（run.sh --debug）でパッチ済みckptのロードと学習再開を確認済み

**結果**（run: nknh13hm、step=142,499→357,500）:
- val/loss: 2.026（再開直後）→ ベスト **1.9795**（step=322,499）。exp029のベスト2.001から約0.02改善し、patience拡大は奏功
- ベスト後28回のvalidationで改善なし、現在1.983（ベスト比+0.003）でプラトー。cosine LRがほぼ0まで減衰し改善余地は縮小
- 過学習の主因はvalue head: val/value_lossはstep=152,499で底（0.5843）を打ち以降緩やかに悪化（last 0.5895）
- policy側は健全: val/policy_lossはstep=322,499まで改善継続（1.4914）、val/policy_accuracyはlast=0.5231が最高
- 所見: 総合val/lossの悪化は小さくvalue過学習が支配的。次の一手はvalue正則化強化（val_lambda調整やweight_decay増）またはベストckpt（step=322,499）での打ち切りが妥当

**結論**:
- 2026-06-13、LRがほぼ0でプラトーに入っていたため手動停止（SIGTERMでgraceful終了）。EarlyStoppingのpatience=50到達を待たず早期に打ち切り
- 成果物（最終ckpt）: `wandb/wcsc36/nknh13hm/checkpoints/epoch=1-step=322500.ckpt`（val/loss=1.9795）。ModelCheckpointのsave_top_k=5で上位5個も保存済み
- 次実験の調整候補: value過学習が早いため、patienceは50より短め（例30）で十分。value正則化を狙うならexp036ベストckptからweight_decayを0.01に戻して短期FT

### exp035: Symmetry Consistency Loss（exp029ベストckptから継続）
**日付**: 2026-05-26
**ベース実験**: exp029
**パラメータ数**: 86.1M
**改善内容**:
- exp029ベストckpt（step=116250, val/loss=2.001）から継続、flip augあり（50%）、KDなし
- Symmetry Consistency Loss追加（sym_consistency_ratio=0.1、warmup=5000ステップ）
- exp034（exp033ベストckptから継続、val/loss=2.189）との比較実験
- LRスケジュール: t_initial=100000、peak LR=0.0003、warmup_t=1000
- EarlyStopping: patience=30（resume時はpatience=80に拡張、LRが下がる前に停止しないよう調整）
- 現在のベスト val/loss: 2.151（step=7500時点）

### exp034: Symmetry Consistency Loss
**日付**: 2026-05-08
**ベース実験**: exp033
**パラメータ数**: 86.1M
**改善内容**:
- exp033ベストckptから継続、flip augあり（50%）、KDなし
- Symmetry Consistency Loss追加: モデルが盤面とその水平反転に対して一貫した予測を出すよう強制
- Value一貫性: MSE(sigmoid(v_orig), sigmoid(v_flip))
- Policy一貫性: MSE(softmax(p_orig), flip(softmax(p_flip))) — 正しい2187次元ラベルflipテーブルを使用
- sym_consistency_ratio=0.1、warmup=5000ステップで徐々に適用
- 追加フォワードパスが1回増えるため、メモリ・速度に注意（accumulate_grad_batches=8で調整済み）
- LRスケジュール: ファインチューニング用に短縮（t_initial=100000、peak LR=0.0003）

### exp033: ゼロからflip augmentationのみで学習
**日付**: 2026-05-05
**ベース実験**: exp026
**パラメータ数**: 86.1M
**改善内容**:
- exp026（ゼロから学習、flip augなし）と同一アーキテクチャ・設定でflip augを追加
- resume_modelなし（ランダム初期化）、KDなし（kd_ratio=0.0）
- 50%水平反転augmentationを学習開始時から適用
- 目的: flip augを最初から適用した場合の汎化性能向上を検証（exp031はexp029ベストckptからの継続）
- 次実験(exp034): このベストckptからflip augなしでファインチューニング予定

### exp032: exp029ベストckptの構造的プルーニング + KDファインチューニング
**日付**: 2026-04-28
**ベース実験**: exp029
**パラメータ数**: 65.7M（プルーニング後）/ 86.1M（teacher: exp029ベストckpt固定）
**改善内容**:
- exp029ベストckpt（step=116250, val/loss=2.001）にL1ノルムチャネルプルーニングを適用
- 対象: 39個のInceptionNeXtBlockのpwconv1/pwconv2（MLP展開次元 2048→1536、25%削減）
- depthwise conv・attention block・ヘッドはプルーニング対象外
- パラメータ削減: 86.1M→65.7M（-24%）
- ファインチューニング: KD（teacher=exp029ベストckpt、kd_ratio=0.5、temperature=2.0）+ flip aug
- 手順: `bash prune.sh` でpruned_state_dict.ptを生成 → `bash run.sh` でファインチューニング

### exp031: exp029ベストckptからflip augmentationのみで継続学習
**日付**: 2026-04-22
**ベース実験**: exp029
**パラメータ数**: 86.1M
**改善内容**:
- exp029（KD + flip aug）がearly stopしたため、ベストckpt（step=116250, val/loss=2.001）から継続
- KD損失を除去（kd_ratio=0.0）し、flip augmentationのみで学習
- LRコサインスケジュールをリセット（resume_modelで重みのみ引き継ぎ）
- KDなしでflip augの効果を最大限に引き出すことが目的

### exp030: KD only（flip無し）
**日付**: 2026-04-12
**ベース実験**: exp028
**パラメータ数**: 3.8M（student）/ 86.1M（teacher: exp026）
**改善内容**:
- exp028からflip augmentationを除いたアブレーション実験
- KDの効果とflipの効果を切り分けるための比較用
- Policy loss: 50% CE + 50% KD soft target（exp026ベストckpt、temperature=2.0）

### exp029: exp026 (86.1M) + KD + 水平反転データ拡張
**日付**: 2026-04-12
**ベース実験**: exp026
**パラメータ数**: 86.1M（student）/ 86.1M（teacher: exp026ベストckpt固定）
**改善内容**:
- exp026（86.1M）をベースに水平反転augmentation + KD損失を追加
- resume_model: exp026ベストckpt（step=260000）から重みのみ引き継ぎ（LRリセット）
- Policy loss: 50% CE + 50% KD soft target（teacher=exp026ベストckpt固定、temperature=2.0）
- teacherはbf16で保持してメモリ節約
- 動機: exp027で汎化性能向上・policy loss悪化の問題をKDで解決。studentとteacherが同アーキテクチャなのでKDが完全に活用できる

### exp028: Knowledge Distillation + 水平反転データ拡張
**日付**: 2026-04-12
**ベース実験**: exp027
**パラメータ数**: 3.8M（student）/ 86.1M（teacher: exp026）
**改善内容**:
- exp027（水平反転augmentation）をベースに損失関数を変更
- Policy loss: 50% 通常CE loss + 50% KD soft target（exp026ベストckpt: step=260000を教師）
- 教師temperature=2.0でsoft targetを生成（KD loss = CE(y/T, softmax(t/T)) × T²）
- LRコサインスケジュール: cycle_limit=1（1周期のみ、exp026と同設定）
- 動機: exp027で左右反転データ拡張により汎化性能は向上したがpolicy lossが悪化。KDで補正を試みる

### exp027: 水平反転データ拡張
**日付**: 2026-04-02
**ベース実験**: exp022
**パラメータ数**: 3.8M
**改善内容**:
- 訓練データの50%に水平反転（左右反転）augmentationを適用
- 盤面特徴量（features1/features2）のfile軸を反転、指し手ラベルのLEFT↔RIGHT方向を入れ替え
- `dlshogi/augmentation.py`に汎用的なflipルーチンを実装（NumPyルックアップテーブル方式）
- `ptl.py`のModelに`flip_augmentation`/`flip_ratio`パラメータを追加、training_stepで適用
- デバッグ実行（batch=32, 2epoch, GPU）で動作確認済み

### exp026: DropPath + bf16-mixed + grad clip（ゼロから学習）
**日付**: 2026-03-28
**ベース実験**: exp025
**パラメータ数**: 86.1M
**改善内容**:
- exp025と同一アーキテクチャ・設定だが、チェックポイントなしでゼロから学習
- exp025（exp023ベストから継続）が失敗したため、DropPath導入時はゼロからの学習が必要と判断
- DropPath 0→0.2線形スケジュール、bf16-mixed、gradient_clip_val=1.0

### exp025: DropPath + bf16-mixed + gradient clipping ❌失敗
**日付**: 2026-03-28
**ベース実験**: exp023
**パラメータ数**: 86.1M
**改善内容**:
- exp023のベストチェックポイント（val/loss=2.111, step=71250）から継続学習
- DropPath（stochastic depth）を全残差ブロックに追加: 線形スケジュール 0→0.2
- precision: 16-mixed → bf16-mixed（attention overflowの防止、exp021で検証済み）
- gradient_clip_val: 1.0を追加（突然のloss崩壊を防止）
- DropPathは深いブロックほど高い確率でスキップし、構造的過学習を抑制

### exp024: exp023 deep512 + haoデータセットで継続学習 ❌失敗
**日付**: 2026-03-27
**ベース実験**: exp023
**パラメータ数**: 86.1M
**改善内容**:
- exp023のベストチェックポイント（val/loss=2.111, step=71250）からhaoデータセット（287GB, 17ファイル）で継続学習
- OOM回避のため1エポックにつき1ファイルずつ読み込むカスタムHcpeDataModuleを実装
- accumulate_grad_batches=8、batch_size=512、torch.compile有効
- 結果: val/loss=8.342に爆発し完全に失敗。exp023の2.111から大幅に悪化
- 原因: データドメインの違い、またはLRスケジューラのリセットによる学習率の不整合が疑われる

### exp023: InceptionNeXt deep512 + plain attention tail
**日付**: 2026-03-15
**ベース実験**: exp022
**パラメータ数**: 86.1M
**改善内容**:
- exp022のアーキテクチャをスケールアップ: depths=[10]→[40], dims=[192]→[512]
- 構成: 39 InceptionNeXtブロック + 1 plain attentionブロック
- パラメータ数: 3.8M→86.1M（約22.7倍）
- 訓練時batch=1024ではOOM（24GB GPU）、batch=512で動作確認済み
- 推論速度: 165.6ms（batch=128）— exp022の8.2msから約20倍遅い

### exp022: InceptionNeXt + plain self-attention tail block
**日付**: 2026-03-14
**ベース実験**: exp021 / exp015
**パラメータ数**: 3.8M
**改善内容**:
- exp021のSqueezeformerブロックを最小限のplain self-attentionブロックに置換
- 構成: 9 InceptionNeXtブロック + 1 plain attentionブロック（MHA + FFN、pre-norm）
- 位置エンコーディングなし: 9層のconv特徴量が既に空間情報を保持しているため不要と判断
- exp021比でパラメータ数削減（4.3M→3.8M）、RelPosEnc・ConvModule・dual FFNを除去
- softmax attentionによる鋭い注意分布が盤面ゲームに適している（少数の重要マスに集中）
- 推論速度: 8.2ms（exp015: 7.8ms、1.05x遅い、batch=128）— exp021の10.7msから大幅改善

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
