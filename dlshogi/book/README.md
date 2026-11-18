# dlshogi.book prerequisites

PUCT Book Builder stores the book graph in PostgreSQL and builds a small C
extension (`puct_edge`) against the PostgreSQL server headers via PGXS. The
following system packages must be installed before building or running this
package:

- `libpq-dev` — PostgreSQL client library headers, required by `asyncpg`'s
  build (when a wheel is unavailable) and by tooling that links against
  `libpq`.
- `postgresql-server-dev-<N>` — PostgreSQL server headers and the PGXS build
  infrastructure used to build the `dlshogi/book/pgext/puct_edge.c` extension.
  `<N>` must match the major version of the target PostgreSQL server (e.g.
  `postgresql-server-dev-16`).
- PostgreSQL server, version 14 or later — the only supported storage
  backend for the book graph.

Example (Debian/Ubuntu, PostgreSQL 16):

```bash
sudo apt-get install libpq-dev postgresql-server-dev-16 postgresql-16
```

Python dependencies (`asyncpg`, `onnxruntime-gpu`, `numpy`, and the
`pytest` / `pytest-asyncio` / `hypothesis` dev dependencies) are declared in
the repository's `Pipfile`.

## How to run

The CLI is `python -m dlshogi.book` with four subcommands: `import-terashock`
(optional), `search`, `propagate`, and `export`. Run
`python -m dlshogi.book <subcommand> --help` for the full argument list.

### CLI options

| Subcommand | Option | Default | Meaning |
| --- | --- | --- | --- |
| `search` | `--config FILE` | required | Path to the YAML configuration file. |
| `search` | `--model FILE` | required | Path to the ONNX model file. |
| `search` | `--gpu-count N` | 1 | Number of GPU worker processes to spawn (keep ≤ your GPU count). |
| `search` | `--verbose` | off | Verbose progress logging. |
| `search` | `--log-path FILE` | (stdout) | Path for the log output file. |
| `propagate` | `--config FILE` | required | Path to the YAML configuration file. |
| `export` | `--config FILE` | required | Path to the YAML configuration file. |
| `export` | `--apery FILE` | none | Output path for the Apery binary book (omit to skip). |
| `export` | `--yaneuraou FILE` | none | Output path for the YaneuraOu `.db` text book (omit to skip). |
| `import-terashock` | `--config FILE` | required | Path to the YAML configuration file. |

These flags are the only options outside the YAML file; there are no
environment variables or other config mechanisms (the five `PostgreSQL`
values in the YAML are the only way to reach the database).

### 1. Build and install the `puct_edge` extension

```bash
cd dlshogi/book/pgext
make && make install
psql -h <host> -U <user> -d <db> -c "CREATE EXTENSION puct_edge;"
```

`make` uses the server's `pg_config`, so `postgresql-server-dev-<N>` must
match the target server's major version.

### 2. Write the config file

The YAML keys are the exact Requirement-13 names (see `config.py`'s
`RANGE_TABLE`). Every one is required except `Terashock_Book path`. Example:

```yaml
PostgreSQL host: localhost
PostgreSQL port: "5432"          # a string, not an int
PostgreSQL user: operator
PostgreSQL password: "secret"
PostgreSQL database: puct_book
Cache_Budget: 268435456          # bytes, >= 256 MiB
Worker_Count: 1024
Batch_Size: 128
Batch_Timeout: 200               # milliseconds
Virtual_Loss: 3
Terashock_Prior_Weight: 0.5
Eval_Coef: 600
Draw_Value_Black: 0.5
Draw_Value_White: 0.5
Max_Book_Ply: 256
Propagation_Visit_Threshold: 100
Export_Visit_Threshold: 0.01
Connection_Retry_Limit: 5
Root_Position: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
Report_Interval: 60
Throughput_Floor: 50
Throughput_Grace_Period: 600
# Terashock_Book path: /path/to/terashock.db   # optional
```

If any required key is missing or out of range, validation reports **every**
offending name and exits with status 1 rather than starting partially.

Three cross-value sizing checks (e.g. `Worker_Count < 2 * Batch_Size`,
`Batch_Size > 2 * Throughput_Floor * Batch_Timeout`, `Throughput_Floor`
above the measured per-process descent-rate ceiling) emit **warnings** at
startup; they never block the run. `PostgreSQL password` is the only
credential and is redacted in all startup logs.

### Configuration options

All ranges are inclusive. "required" = must be present (only
`Terashock_Book path` is optional).

| Key | Range / unit | Meaning |
| --- | --- | --- |
| `PostgreSQL host` | 1–1,024 chars | Book-graph database server host. |
| `PostgreSQL port` | 1–1,024 chars (**string**, not int) | Database server port. |
| `PostgreSQL user` | 1–1,024 chars | Database user. |
| `PostgreSQL password` | 1–1,024 chars | Database password; redacted in logs. |
| `PostgreSQL database` | 1–1,024 chars | Database name. |
| `Cache_Budget` | 256 MiB – 1 TiB, bytes | Byte budget for the in-process node LRU. The LRU never exceeds this size and under pressure it triggers out-of-band flushes (flush threshold is half the budget); a share of it is also reserved for the Terashock index LRU. |
| `Worker_Count` | 1–1,024 | Number of concurrent search workers in `search`. |
| `Batch_Size` | 1–4,096 | Fixed Evaluator batch size: positions are collected up to this size, short batches are zero-padded to it. |
| `Batch_Timeout` | 1–1,000 ms | Max time to wait for a full `Batch_Size` before dispatching the partial batch to the Evaluator. |
| `Virtual_Loss` | 0–16 | Visits added to in-flight child nodes during parallel PUCT search so concurrent workers do not collapse onto the same move. |
| `Terashock_Prior_Weight` | 0.0–1.0 | Mix weight of the Terashock-derived prior against the network policy (0 = pure network, 1 = pure Terashock). |
| `Eval_Coef` | 1–10,000 | Temperature of the sigmoid that turns a Terashock evaluation into a win rate: `1 / (1 + exp(-score / Eval_Coef))`. Also used to convert exported values back to log-odds. |
| `Draw_Value_Black` | 0.0–1.0 | Score assigned to a draw for Black (side to move). 0.5 is an exact draw. |
| `Draw_Value_White` | 0.0–1.0 | Same for White. |
| `Max_Book_Ply` | 0–1,024 | Depth limit for book growth in `search` and frame partitioning in `propagate`. 0 disables the depth limit. |
| `Propagation_Visit_Threshold` | 0–1,000,000 (absolute visits) | Edges with fewer visits are handled in a separate (lighter) partition during the negamax propagation pass. 0 disables the partition. |
| `Export_Visit_Threshold` | 0.0–1.0 (ratio) | Edges whose `visit_count / node_visit_count` ratio is below this are excluded from the exported book. |
| `Connection_Retry_Limit` | 0–100 | Number of PostgreSQL reconnect retries after a failed attempt, with waits of 1 s, 2 s, 4 s, … capped at 60 s. |
| `Root_Position` | 1–256 chars (SFEN) | Root position of the book; `search` grows the graph from here and `import-terashock` seeds it here. SFEN legality is checked at run time, not at config validation. |
| `Report_Interval` | 1–3,600 s | Seconds between progress-report ticks. |
| `Throughput_Floor` | 0.01–100,000 descents/s | Measured descent rate below which the degraded-throughput warning may fire. Keep it at or below roughly 1,500 × (number of `search` processes). |
| `Throughput_Grace_Period` | 60–86,400 s | How long the rate may stay below `Throughput_Floor` before the warning fires. |
| `Terashock_Book path` | 1–4,096 chars, **optional** | Path to a YaneuraOu Terashock `.db` to seed from. If absent, `import-terashock` and the Terashock prior index are disabled. |

### 3. Build the book
The convenience launcher `puct-book` runs the same pipeline with optional `SEARCH_SECONDS` (default 2,592,000 seconds / 30 days) and `GPU_COUNT` (default 2). It stops Strata and llama-swap when GPU preflight finds insufficient free memory, then restores services that were running before the launch. A manual Ctrl-C stops search without running propagation or export; run those steps afterward if needed. For a non-destructive GPU check, use `PREFLIGHT_ONLY=1 bash dlshogi/book/run.sh` directly.

Use `puct-bookctl start [SEARCH_SECONDS] [GPU_COUNT]` for detached book building. Use `puct-bookctl restart [SEARCH_SECONDS] [GPU_COUNT]` to stop the tracked builder gracefully and start it again. Use `puct-bookctl status` to monitor it; status exits 0 only when the tracked builder is running and 1 when the builder is stopped, its PID metadata is stale, or the controller is busy. `puct-bookctl stop` sends a graceful interrupt and does not force-kill the builder; propagation and export run only when the search reaches its normal timed stop. The controller keeps its state in `output/.puct-book`.

Use `puct-bookctl book [SEARCH_SECONDS] [GPU_COUNT]` to switch to book mode. It validates its arguments and service scripts, records which of Strata and llama-swap are running, stops both services, and starts the builder. The record is stored at `output/.puct-book/restore-services`; if book mode is already in progress, it is kept unchanged. A failed restore also keeps the record, while a successful restore or successful `puct-bookctl llama` removes it. `puct-bookctl restore` refuses while a builder holds the run lock, so manual use is `puct-bookctl stop` followed by `puct-bookctl restore`; the watcher runs `stop` first. Restore starts Strata before llama-swap because `strata.sh start` stops the llama-swap container to free the GPUs. Restore requires `HOME` and both service launchers, except for an empty record, and can block for about 10 minutes while the Strata model loads; during that time `status` reports that the controller is busy, the watcher's next ticks are skipped, and other `puct-bookctl` commands wait for the lock. The record covers only the orca-path Strata server, not the sibling `~/opt/Strata` install. If a Strata start times out, the next attempt is refused with "a strata process exists but the API is down"; run `~/opt/strata-orca/strata.sh restart` by hand, and the watcher will then finish the restore. If you stop the build early, run `propagate` and `export` manually before switching modes because the interrupt path does not run them. `run.sh` supports `PREFLIGHT_ONLY=1` only when invoked directly, and exits without its own service shutdown.
A cron watcher can automate this handoff when the Windows host at `192.168.1.110` changes state. `dlshogi/book/windows_book_watch.sh` checks it every minute, waits 10 minutes of continuous reachability loss before starting book mode, and restores the pre-book model services when the host returns; the stop may take up to the builder's graceful-shutdown timeout. A brief reboot or network interruption therefore does not start a build. The watcher stores its state and log under `output/.puct-book/` and serializes ticks with a lock.

```bash
# grow the book graph with PUCT search (spawns --gpu-count GPU worker processes)
python -m dlshogi.book search --config config.yaml --model path/to/model.onnx --gpu-count 2

# run the negamax value-propagation pass
python -m dlshogi.book propagate --config config.yaml

# export to Apery and/or YaneuraOu
python -m dlshogi.book export --config config.yaml --apery book.ap --yaneuraou book.db
```

Optionally seed from an existing YaneuraOu Terashock book before `search`:

```bash
python -m dlshogi.book import-terashock --config config.yaml
```

`search` is the only GPU-bound step (keep `--gpu-count` at or below your GPU
count); `propagate` and `export` use the database only.
