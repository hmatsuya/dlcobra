# 予測エラー分析 ベストプラクティス

調査日: 2026-04-04

## 1. Confusion Matrix（誤分類パターン）

- クラス間の誤分類パターンを把握
- どのクラスが何に間違えられているか特定
- dlcobra適用例：正解手 vs 予測手の方向・距離分布、駒種別の誤り率

## 2. Grad-CAM / Grad-CAM++（注目領域の可視化）

- 最終畳み込み層の勾配からヒートマップ生成
- モデルが「どこを見て」判断したか可視化
- dlcobra適用例：9×9盤面のどのマスに注目して指し手を選んだか可視化

```python
# PyTorch Grad-CAM 基本実装
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

cam = GradCAM(model=model, target_layers=[model.last_conv_layer])
grayscale_cam = cam(input_tensor=input, targets=[ClassifierOutputTarget(target_class)])
```

参考: [Grad-CAM論文](https://arxiv.org/abs/1610.02391)

## 3. Top-K エラー分析

- 確信度が高いのに間違えたサンプルを優先調査
- 「簡単なはずなのに間違えた」ケースが設計上の問題を示す
- 実装：policy出力のtop-1予測が不正解かつsoftmax確率が高いサンプルを抽出

## 4. スライス分析（Slice Analysis）

全体精度ではなくサブグループ別にエラー率を分離する。

dlcobra向けスライス例：
| スライス | 分類方法 |
|---------|---------|
| 序盤/中盤/終盤 | 手数で分割 |
| 駒得/駒損局面 | 評価値で分割 |
| 王手局面 | 王手フラグで分割 |
| 持ち駒多/少 | 持ち駒数で分割 |

特定局面タイプで精度が低い場合 → そのデータを重点的に増強

## 5. Hard Example Mining

- 繰り返し間違えるサンプルを収集・学習データに追加
- Focal Lossはこれの自動化（exp025で導入済み）
- 手動版：validation誤りサンプルをログに記録し次回学習に追加

## 6. Calibration（信頼度校正）

- Policy head: softmax出力が過信気味なら温度スケーリングで補正
- Value head: 勝率予測精度はECE（Expected Calibration Error）で測定

```python
# 温度スケーリング
logits_calibrated = logits / temperature  # temperature > 1 で分布を平坦化
```

## dlcobra 実装優先度（推奨順）

1. **スライス分析** — 既存のvalidationデータで即実施可能。局面タイプ別policy accuracyを計測
2. **Top-K エラー分析** — 誤り局面をSFEN形式で保存し、将棋GUIで確認
3. **Grad-CAM** — 盤面の9×9ヒートマップで注目マス可視化
4. **Calibration** — Value headの勝率予測精度改善
