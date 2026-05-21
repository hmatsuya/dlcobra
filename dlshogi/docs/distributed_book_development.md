# Distributed MCTS Book Development

## Overview

PostgreSQLベースのハイブリッドアプローチで、ブラウザ（WebGPU）を活用した分散定跡開発を行う構想。

## Architecture

### Hybrid Approach

- **In-memory MCTS**: 各ワーカーがバッチ単位（例: 1M nodes）でMCTS探索を実行
- **PostgreSQL**: 永続化・重複排除・分析用のストレージ
- メモリが溢れたらDBにフラッシュし、DBの状態から再開

```
┌─────────────────────────────────────────┐
│  Browser (volunteer contributor)         │
│  ┌───────────┐  ┌────────────────────┐  │
│  │ WebGPU    │  │ MCTS (TypeScript)  │  │
│  │ NN infer  │  │ shogiops           │  │
│  └─────┬─────┘  └──────┬─────────────┘  │
│        └────────────────┘                │
│              │ batch results             │
└──────────────┼───────────────────────────┘
               │ HTTPS POST
        ┌──────▼──────┐
        │  API Server  │
        └──────┬──────┘
               │ batch upsert
        ┌──────▼──────┐
        │  PostgreSQL  │  (Supabase)
        └─────────────┘
```

### Database Schema

```sql
CREATE TABLE mcts_nodes (
    position_hash BIGINT PRIMARY KEY,
    visit_count INT,
    value_sum FLOAT,
    policy FLOAT[],
    children BIGINT[]
);
```

### Merge Strategy (Distributed Upsert)

```sql
INSERT INTO mcts_nodes (position_hash, visit_count, value_sum, policy)
VALUES ($1, $2, $3, $4)
ON CONFLICT (position_hash) DO UPDATE SET
  visit_count = mcts_nodes.visit_count + EXCLUDED.visit_count,
  value_sum = mcts_nodes.value_sum + EXCLUDED.value_sum;
```

## Technology Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| Database | PostgreSQL (Supabase) | 管理不要、分散アクセス、SQL分析、スケール時は自前PGに移行可 |
| NN推論 (Browser) | ONNX Runtime Web + WebGPU | モデルをONNX exportして配信 |
| 合法手生成 (Browser) | [shogiops](https://github.com/WandererXII/shogiops) | TypeScript、perft検証済み、lishogi.orgで実績あり |
| MCTS実装 (Browser) | TypeScript + Web Workers | ブラウザ内並列処理 |

### shogiops選定理由

- lishogi.org（将棋版lichess）で本番利用されている
- 完全な合法手生成（perft tested）
- SFEN対応
- 純TypeScript（WASM不要）
- GPL3ライセンス → dlshogi（GPL3）と互換、ブラウザクライアント公開に問題なし

### Shogi.js (na2hiro) を選ばなかった理由

- UI/棋譜再生向けで、パフォーマンス重視の手生成ではない
- SFEN非対応
- エッジケース（打ち歩詰め等）の検証が不十分な可能性

## Performance Estimates

| Environment | Simulations/sec |
|-------------|-----------------|
| Native CUDA (RTX 3080) | 10,000-50,000 |
| WebGPU (discrete GPU) | 500-2,000 |
| WebGPU (integrated GPU) | 100-500 |
| CPU-only (WASM fallback) | 10-50 |

100人のボランティア × 500 sims/sec = 50,000 sims/sec（高性能GPU 1台相当）

## Supabase Considerations

| Scale | 適性 |
|-------|------|
| プロトタイプ / 小規模 (<50M nodes) | ✅ Free tierで十分 |
| 中規模、2-5 workers | ✅ Pro planで対応可 |
| 大規模 (100M+ nodes) | ⚠️ コスト増、自前PostgreSQLへ移行推奨 |

移行は `pg_dump` / `pg_restore` で容易（コード変更不要）。

## Implementation Steps

1. モデルをONNX exportする（`convert_model_to_onnx.py`既存）
2. ブラウザクライアント: ONNX Runtime Web (WebGPU) + shogiops + MCTS
3. APIサーバー: バッチ結果を受信、バリデーション、Supabaseへupsert
4. タスクサーバー: フロンティア局面をクライアントに割り当て
5. Web UI: 貢献状況の可視化

## Trust & Validation

- サーバー側でスポットチェック（局面の合法性確認）
- レートリミット per user
- `model_version` カラムで異なるモデルバージョンの結果を区別
- ワーカークラッシュ時: 未フラッシュ分は消失するがDB整合性は維持
