# 定跡ファイル (Book File) ガイド

## ファイル形式

定跡ファイルはバイナリ形式 (`.bin`)。各エントリは以下の構造体 (`cppshogi/book.hpp`):

```cpp
struct BookEntry {
    Key    key;        // 局面ハッシュ (Zobristハッシュ)
    u16    fromToPro;  // 指し手 (from/to/成り)
    u16    count;      // 出現回数
    Score  score;      // 評価値
};
```

エントリは `key` の昇順でソートされており、局面検索はバイナリサーチで行われる。

---

## 対局中の USI オプション

`OwnBook` を有効にすると、`go` コマンド受信時に定跡を参照する。

| オプション | デフォルト | 説明 |
|---|---|---|
| `OwnBook` | `false` | 定跡を使用するか |
| `Book_File` | `book.bin` | 定跡ファイルのパス |
| `Best_Book_Move` | `true` | `true` = 最多出現手を選択、`false` = 出現頻度に比例してランダム選択 |
| `Min_Book_Score` | `-3000` | この評価値未満の定跡手はスキップ |
| `Max_Book_Ply` | `0` | この手数以降は定跡を使用しない (0=無制限) |
| `Book_Consider_Draw` | `false` | 千日手の評価値を考慮して手を選ぶ |
| `Book_Consider_Draw_Depth` | `0` | 何手先まで千日手を読むか (0=直前のみ、>0=depth手先まで再帰) |

`Book_Consider_Draw_Depth > 0` のとき、`Draw_Value_Black` / `Draw_Value_White` / `Eval_Coef` も間接的に影響する。

### 設定例

```
setoption name OwnBook value true
setoption name Book_File value /path/to/book.bin
setoption name Best_Book_Move value false      # 多様な手を指す場合
setoption name Min_Book_Score value -1000      # 低評価の手を除外
setoption name Book_Consider_Draw value true
setoption name Book_Consider_Draw_Depth value 3
isready
```

---

## 探索ロジック

`probe()` の動作:
1. 局面ハッシュでバイナリサーチ
2. マッチするエントリを走査
   - `Best_Book_Move=true` → `count` 最大の手を選択
   - `Best_Book_Move=false` → `count` に比例した確率でランダム選択
   - `Min_Book_Score` 未満の評価値の手はスキップ
3. 定跡にない局面、または `Max_Book_Ply` を超えた手数の場合は通常のMCTS探索にフォールバック

`Book_Consider_Draw=true` のとき:
- 千日手 → `Draw_Value_Black` / `Draw_Value_White` で評価値を上書き
- 相手の勝ち (反復勝ち) → `-ScoreInfinite`
- 自分の勝ち (反復負け) → `ScoreMaxEvaluate`

---

## 定跡ファイルの作成

### 方法1: CSA棋譜から作成

USIコマンドで実行:

```
make_book <棋譜ファイル>
```

- 勝った側の手のみを記録
- 出現回数がそのまま選択確率になる
- 出力: `book.bin`

棋譜フォーマット (1行目: ヘッダ、2行目: CSA形式の指し手列):

```
1 2003/09/08 羽生善治 谷川浩司 2 126 王位戦 その他の戦型
7776FU3334FU2726FU4132KI...
```

### 方法2: UCT探索で自動生成

```
make_book <既存定跡ファイル or 空ファイル>
```

関連オプション (make_book 専用、対局中は不使用):

| オプション | デフォルト | 説明 |
|---|---|---|
| `Use_Book_Policy` | `false` | ポリシーネットワークを定跡生成に使用 |
| `Book_Eval_Threshold` | `INT_MAX` | 評価値がこれ以上の手のみ採用 |
| `Book_Visit_Threshold` | `5` (=0.5%) | 訪問率がこれ未満の手を除外 |
| `Book_Cutoff` | `15` (=1.5%) | 最善手との差がこれ以上の手を除外 |
| `Book_Temperature` | `1000` | 手選択の温度 (高いほどランダム) |
| `Make_Book_Color` | `"both"` | `"black"` / `"white"` / `"both"` |
| `Book_Merge_File` | `""` | マージする既存定跡ファイル |
| `Save_Book_Interval` | `100` | 何試行ごとに中間保存するか |
| `Make_Book_Sleep` | `0` ms | 試行間のスリープ時間 |

---

## 関連ソースファイル

| ファイル | 内容 |
|---|---|
| `cppshogi/book.hpp` | `BookEntry` 構造体、`Book` クラス定義 |
| `cppshogi/book.cpp` | `probe()`, `makeBook()` 実装 |
| `cppshogi/usi.cpp` | USIオプションのデフォルト値定義 |
| `usi/main.cpp` | `go` コマンドでの定跡参照、`make_book` コマンド実装 |
