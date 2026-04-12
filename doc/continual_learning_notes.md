# Continual Learning（追加学習）調査メモ

作成日: 2026-04-06

---

## 1. 核心問題: Catastrophic Forgetting

新しいデータで学習すると、古いデータへの精度が壊滅的に落ちる。これが最大の課題。

**Joint Training（全データ混合で一から学習）が精度の上限**。Continual Learning の目標はこれに近い精度を、データを混ぜずに達成すること。

---

## 2. アプローチ分類

### Regularization 系

**EWC（Elastic Weight Consolidation）**
- 古いタスクで重要だった重みを「固定」する正則化項を追加
- Fisher 情報行列で重みの重要度を計算
- シンプルだが、タスク数が増えると効果が薄れる。Replay 系より弱い傾向

```python
loss = task_loss + λ * Σ F_i * (θ_i - θ*_i)²
# F_i: Fisher情報（重みの重要度）, θ*: 旧タスクの最適重み
```

### Replay 系（実用上最強）

**Experience Replay**
- 古いデータの一部をバッファに保存し、新データと混ぜて学習
- シンプルで効果が高い
- 欠点: 古いデータを保持する必要がある

**Generative Replay**
- 生成モデル（Diffusion等）で古いデータの合成データを生成して代替
- 古いデータ不要だが、生成品質に依存

### Knowledge Distillation 系

**LwF（Learning without Forgetting）**
- 古いモデルを Teacher として、新学習中に古い出力を蒸留
- 古いデータ不要。新旧タスクが似ているほど効果的

```python
loss = 新タスクの損失 + α * KL(旧モデル出力, 現モデル出力)
```

### Parameter Isolation 系

**LoRA / Adapter（最近の主流）**
- 事前学習済みモデルの重みを凍結し、小さな追加モジュールだけ学習
- 忘却ゼロ、軽量。大規模モデルに特に有効

---

## 3. 実用的な選択基準

| 状況 | 推奨手法 |
|------|---------|
| 古いデータが使える | Experience Replay（混合学習）が最強 |
| 古いデータが使えない | LwF（蒸留）or EWC |
| モデルを大きくしてよい | Adapter / LoRA |
| 精度上限を知りたい | Joint Training（全データ混合）で比較 |

**注意**: Sequential Fine-tuning（新データだけで fine-tune）は古いデータの精度が大幅低下するため危険。最低でも LwF か Experience Replay を組み合わせること。

---

## 4. dlcobra への応用

### データ構造

| データ種別 | 内容 | 特性 |
|-----------|------|------|
| 自己対局データ（HCPE3） | MCTS訪問回数付き棋譜 | 高品質、量が多い |
| 人間棋譜（HCPE） | プロ・アマの実戦棋譜 | 多様な戦型 |
| 教師データ（teacher） | 強エンジンの評価値付き棋譜 | 精度高い |

### 推奨: Experience Replay（混合学習）

将棋AIの場合、「古いデータを忘れる」問題より「新旧データのバランス」が本質的な課題。

```python
# サンプリング比率で新旧データを制御
sampler = WeightedRandomSampler(
    weights=[0.3] * len(old_data) + [0.7] * len(new_data),
    num_samples=total_samples
)
```

### LwF（蒸留）の応用（オプション）

```python
old_policy, old_value = old_model(x)
new_policy, new_value = new_model(x)
distill_loss = KL(old_policy, new_policy)
loss = task_loss + α * distill_loss
```

### シナリオ別推奨

| シナリオ | 推奨手法 |
|---------|---------|
| 自己対局データ → 人間棋譜を追加 | 混合学習（比率調整） |
| 古いモデルに新しい戦型データを追加 | LwF + 混合学習 |
| 全データで再学習したい | Joint Training（全混合）が上限 |
| 計算コストを抑えたい | 新データのみ fine-tune + EWC |

### AlphaZero 系の実例

AlphaZero 自体は「リプレイバッファ」方式（FIFO）で Continual Learning を実現している。dlcobra でも新旧データの混合比率をコントロールするのが最もシンプルで効果的。

### 実装の起点

`fine_tuning.py` に以下を追加するのが最短経路：
1. 新旧データの混合比率パラメータ
2. 旧モデルを Teacher とした蒸留損失（オプション）
3. WandB で新旧データ別の精度を追跡

---

## 参考文献

- [Continual Learning with Knowledge Distillation: A Survey](https://www.researchgate.net/publication/377097224_Continual_Learning_with_Knowledge_Distillation_A_Survey)
- [Recent Advances of Continual Learning in Computer Vision](https://arxiv.org/html/2109.11369)
- [Unleash the Power of Sequential Fine-tuning for Continual Learning](https://arxiv.org/html/2408.08295)

---

## 5. LwF + Experience Replay の組み合わせ

### それぞれの弱点と補完関係

| 手法 | 弱点 |
|------|------|
| LwF のみ | 新旧タスクが大きく異なると蒸留が機能しにくい |
| Replay のみ | 古いデータの保存量が限られると forgetting が残る |

組み合わせると両方の弱点を補える。iCaRL などで使われているアプローチに近い。

### 損失関数の加算について

**加算は問題ない**。`loss_task` と `loss_replay` はそれぞれ独立したサンプルから計算した勾配を足し合わせているだけで、マルチタスク学習・混合バッチと本質的に同じ。

```python
# 実装上は「混合バッチ」にするのが最もシンプル
batch = concat(new_data_batch, old_data_batch)
loss = criterion(model(batch), labels)
```

### 注意点

**スケールの問題**: バッチサイズが違うと損失の大きさが不均衡になる。

```python
# NG: バッチサイズが違うと勾配の大きさが変わる
loss = loss_new(n=256) + loss_old(n=64)

# OK: 平均を取る（バッチサイズで正規化）
loss = loss_new.mean() + α * loss_old.mean()
```

**LwF の蒸留は同じ局面に対して計算する必要がある**:

```python
# NG: 違う局面で比較しても意味がない
loss_distill = KL(old_model(new_x), new_model(old_x))

# OK: 同じ x を両モデルに通す
loss_distill = KL(old_model(x).detach(), new_model(x))
```

### dlcobra での実装イメージ

```python
# ミニバッチ = 新データ + 古いデータを混合
x = concat(new_x, old_x)
y = concat(new_y, old_y)

# 通常の学習損失（新旧混合バッチ）
policy_pred, value_pred = new_model(x)
loss_task = policy_loss(policy_pred, y) + value_loss(value_pred, y)

# LwF: 同じバッチを旧モデルにも通す
with torch.no_grad():
    old_policy, _ = old_model(x)
loss_distill = KL(old_policy, policy_pred)

loss = loss_task + β * loss_distill
```

α（リプレイ比率）と β（蒸留強度）は WandB で新旧データ別の精度を見ながらチューニングする。
