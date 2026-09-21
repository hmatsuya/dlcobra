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

### 3. Build the book

```bash
# grow the book graph with PUCT search (spawns --gpu-count GPU worker processes)
python -m dlshogi.book search --config book.yaml --model path/to/model.onnx --gpu-count 2

# run the negamax value-propagation pass
python -m dlshogi.book propagate --config book.yaml

# export to Apery and/or YaneuraOu
python -m dlshogi.book export --config book.yaml --apery book.ap --yaneuraou book.db
```

Optionally seed from an existing YaneuraOu Terashock book before `search`:

```bash
python -m dlshogi.book import-terashock --config book.yaml
```

`search` is the only GPU-bound step (keep `--gpu-count` at or below your GPU
count); `propagate` and `export` use the database only.
