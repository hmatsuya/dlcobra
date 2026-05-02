# USI オプション一覧

ソース: `cppshogi/usi.cpp`, `usi/main.cpp`

---

## 定跡 (Book)

対局中に使用されるオプション。詳細は [BOOK_FILE_GUIDE.md](BOOK_FILE_GUIDE.md) を参照。

| オプション | デフォルト | 説明 |
|---|---|---|
| `OwnBook` | `false` | 定跡を使用するか |
| `Book_File` | `book.bin` | 定跡ファイルのパス |
| `Best_Book_Move` | `true` | `true`=最多出現手、`false`=頻度比例ランダム |
| `Min_Book_Score` | `-3000` | この評価値未満の定跡手はスキップ |
| `Book_Consider_Draw` | `false` | 千日手の評価値を考慮 |
| `Book_Consider_Draw_Depth` | `0` | 千日手を読む深さ (0=直前のみ) |

---

## モデル・推論

| オプション | デフォルト | 範囲 | 説明 |
|---|---|---|---|
| `DNN_Model` | `model.onnx` | — | GPU1のモデルファイルパス |
| `DNN_Model2`〜`16` | `""` | — | GPU2〜16のモデルパス |
| `DNN_Batch_Size` | `128` | 1–256 | GPU1のバッチサイズ |
| `DNN_Batch_Size2`〜`16` | `0` | 0–256 | GPU2〜16のバッチサイズ |
| `Softmax_Temperature` | `174` | 1–500 | ポリシーのsoftmax温度 (実値=÷100) |

---

## 探索

| オプション | デフォルト | 範囲 | 説明 |
|---|---|---|---|
| `UCT_Threads` | `2` | 0–256 | GPU1のUCT探索スレッド数 |
| `UCT_Threads2`〜`16` | `0` | 0–256 | GPU2〜16のスレッド数 |
| `UCT_NodeLimit` | `10,000,000` | 100K–1B | UCTノード数の上限 |
| `Const_Playout` | `0` | 0–∞ | 固定プレイアウト数 (0=時間制御) |
| `ReuseSubtree` | `true` | bool | 前の探索木を再利用するか |
| `MultiPV` | `1` | 1–2186 | 複数候補手の出力数 |
| `PV_Interval` | `500` | 0–∞ | PV出力間隔 (ms) |

### PUCT パラメータ

| オプション | デフォルト | 説明 |
|---|---|---|
| `C_init` | `144` (=1.44) | PUCT定数 (÷100) |
| `C_base` | `28288` | PUCT base |
| `C_fpu_reduction` | `27` (=0.27) | FPU reduction (÷100) |
| `C_init_root` | `116` | ルートノードのC_init |
| `C_base_root` | `25617` | ルートノードのC_base |
| `C_fpu_reduction_root` | `0` | ルートノードのFPU reduction |

---

## 時間管理

| オプション | デフォルト | 説明 |
|---|---|---|
| `Byoyomi_Margin` | `0` ms | 秒読み時間から差し引くマージン |
| `Time_Margin` | `1000` ms | 持ち時間から差し引くマージン |

---

## 投了・引き分け

| オプション | デフォルト | 範囲 | 説明 |
|---|---|---|---|
| `Resign_Threshold` | `10` | 0–1000 | 投了閾値 (勝率×1000、10=1%) |
| `Draw_Ply` | `0` | 0–∞ | この手数以降は引き分け扱い (0=無効) |
| `Draw_Value_Black` | `500` | 0–1000 | 先手にとっての引き分けの価値 (×0.001) |
| `Draw_Value_White` | `500` | 0–1000 | 後手にとっての引き分けの価値 |
| `Eval_Coef` | `756` | 1–10000 | 評価値変換係数 |

---

## 詰み探索

| オプション | デフォルト | 範囲 | 説明 |
|---|---|---|---|
| `Mate_Root_Search` | `33` | 0–37 | ルートでのdf-pn詰み探索深さ (0=無効) |
| `DfPn_Hash` | `2048` MB | 64–4096 | df-pnハッシュサイズ |
| `DfPn_Min_Search_Millisecs` | `300` ms | 0–∞ | df-pn最小探索時間 |
| `PV_Mate_Search_Threads` | `0` | 0–256 | PV詰み探索スレッド数 (0=無効) |
| `PV_Mate_Search_Depth` | `33` | 0–37 | PV詰み探索深さ |
| `PV_Mate_Search_Nodes` | `500,000` | 0–10M | PV詰み探索ノード数 |

---

## ランダム手 (序盤多様化)

2系統のランダム手機能がある。

### Random (系統1): 序盤の固定手数

| オプション | デフォルト | 説明 |
|---|---|---|
| `Random_Ply` | `0` | この手数まではランダム手を指す (0=無効) |
| `Random_Temperature` | `10000` (=10.0) | ランダム手の温度 (÷1000) |
| `Random_Temperature_Drop` | `1000` (=1.0) | ランダム手の温度 (drop後) |
| `Random_Cutoff` | `15` (=1.5%) | 最善手との差がこれ以上の手を除外 |
| `Random_Cutoff_Drop` | `0` | Cutoff (drop後) |

### Random2 (系統2): 確率的ランダム手

| オプション | デフォルト | 説明 |
|---|---|---|
| `Random2_Ply` | `0` | この手数まで適用 (0=無効) |
| `Random2_Probability` | `40` (=4%) | ランダム手を選ぶ確率 (÷1000) |
| `Random2_Temperature` | `10000` (=10.0) | 温度 (÷1000) |
| `Random2_Cutoff` | `30` (=3%) | cutoff (÷1000) |
| `Random2_Value_Limit` | `750` (=75%) | 勝率がこれ以上の局面ではランダム手を使わない |

---

## ポンダー

| オプション | デフォルト | 説明 |
|---|---|---|
| `USI_Ponder` | `false` | ポンダーを有効化 |
| `Stochastic_Ponder` | `true` | 確率的ポンダー (相手の手を予測せず探索継続) |

---

## Multi-Ponder (複数エンジン連携)

| オプション | デフォルト | 説明 |
|---|---|---|
| `Multi_Ponder` | `0` | サブエンジン数 (0–9) |
| `Multi_Ponder_Engine1`〜`9` | `""` | サブエンジンの実行パス |
| `Multi_Ponder_Engine1_Options`〜`9` | `""` | サブエンジンのオプション文字列 |

---

## その他

| オプション | デフォルト | 説明 |
|---|---|---|
| `DebugMessage` | `false` | デバッグメッセージ出力 |
| `Engine_Name` | `dlcobra wcsc36` | エンジン名 |

---

## 設定例

### 本番対局 (強さ重視)

```
setoption name UCT_Threads value 4
setoption name DNN_Batch_Size value 128
setoption name Resign_Threshold value 50
setoption name Mate_Root_Search value 33
setoption name OwnBook value true
setoption name Book_File value book.bin
```

### 自己対局 (多様性重視)

```
setoption name Random_Ply value 10
setoption name Random_Temperature value 5000
setoption name OwnBook value true
setoption name Best_Book_Move value false
```

### 引き分け回避 (先手側)

```
setoption name Draw_Value_Black value 300
setoption name Draw_Value_White value 700
setoption name Book_Consider_Draw value true
setoption name Book_Consider_Draw_Depth value 3
```
