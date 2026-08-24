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
