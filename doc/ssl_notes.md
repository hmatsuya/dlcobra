# Self-Supervised Learning (SSL) 調査メモ

作成日: 2026-04-06

---

## 1. SSL と Self-Distillation の関係

```
Self-Supervised Learning（大）
├── Contrastive Learning（SimCLR, MoCo）
├── Masked Image Modeling（MAE, BEiT）
├── Self-Distillation ← DINO, BYOL, SimSiam
└── Predictive（回転予測など）
```

- **SSL**: ラベルなしで学習する手法の総称。教師信号をデータ自体から作る
- **Self-Distillation**: SSL の一手法。同一アーキテクチャの Teacher-Student 構造で、Teacher は自分自身（または EMA コピー）

---

## 2. 画像モデルにおける Self-Distillation ベストプラクティス

### 核心アーキテクチャ: Student-Teacher with EMA

```
Teacher weights = momentum × Teacher + (1 - momentum) × Student
```

Teacher は直接学習せず、EMA で Student から更新。

### 主要テクニック

| テクニック | 内容 |
|-----------|------|
| Multi-crop augmentation | Global view × 2 → Teacher/Student 両方。Local view × 複数 → Student のみ。「local-to-global」対応を学習 |
| Centering + Sharpening | Teacher 出力に centering（バッチ統計で引き算）。Softmax temperature を Teacher 側で低く設定。これがないと mode collapse |
| EMA momentum スケジューリング | 初期 0.996 → 後期 0.9999（cosine schedule） |
| Random Masking（DINOv2以降） | ViT の patch を一部マスクして学習効率向上 |

### よくある失敗と対策

| 問題 | 原因 | 対策 |
|------|------|------|
| Mode collapse | centering/sharpening 不足 | temperature と centering を必ず入れる |
| 学習不安定 | momentum が低すぎ | EMA momentum ≥ 0.99 から始める |
| 表現が弱い | augmentation が弱い | RandAugment + color jitter + blur を強めに |

---

## 3. 主要フレームワークの系譜

| 手法 | 特徴 |
|------|------|
| DINO (2021) | ViT + self-distillation の原点。ラベルなしで強力な特徴量 |
| DINOv2 (2023) | 大規模データ + curated dataset + masked image modeling 追加 |
| BYOL | Negative pair 不要。Projector + Predictor の非対称構造 |
| SimSiam | EMA なしでも崩壊しない（stop-gradient が鍵） |

---

## 4. CNN vs ViT

**CNN でも動く**。BYOL・SimSiam・MoCo は元々 ResNet ベース。

ViT の方が優れる理由（DINO 論文の発見）:
- attention map が自然にセグメンテーションマップになる（emergent property）
- k-NN 分類器として使ったとき ViT の特徴量が明らかに強い
- CNN は局所的な畳み込みが基本なので、グローバルな意味的構造が出にくい

### 実用的な選択基準

| 状況 | 推奨 |
|------|------|
| リソース潤沢、表現品質最優先 | ViT + DINO/DINOv2 |
| 軽量・エッジデバイス | CNN + BYOL or SimSiam |
| セグメンテーション・検出の特徴量として使う | ViT 一択 |

---

## 5. Kaggle での活用

### 主な使われ方

1. **事前学習済みモデルの利用（最多）**: DINOv2、CLIP、MAE のバックボーンをそのまま使う
2. **ドメイン適応 pretraining**: コンペの test データ含むラベルなし画像で SSL 追加学習 → fine-tune（Kaggle では合法）
3. **Pseudo Labeling との組み合わせ**: SSL で特徴量抽出 → k-NN/clustering で pseudo label → supervised fine-tune

### コンペタイプ別

| コンペタイプ | SSL の使い方 |
|------------|------------|
| 医療画像（病理、X線） | DINO pretrain → linear probe |
| ラベル少ない分類 | SSL pretraining + pseudo label |
| 衛星・リモートセンシング | ドメイン適応 SSL |
| 一般画像分類 | DINOv2/CLIP backbone をそのまま使う |

**実態**: ゼロから SSL 学習よりも「SSL で学習済みモデルを使う」が圧倒的多数。コンペ期間中にゼロから学習するには計算コストが重すぎる。

---

## 参考文献

- [DINO: Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- [A Cookbook of Self-Supervised Learning](https://arxiv.org/html/2304.12210)
- [Rethinking Random Masking in Self-Distillation on ViT](https://arxiv.org/html/2506.10582v1)
