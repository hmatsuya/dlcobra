"""Pairwise CPU benchmark for the PUCT, Terashock, and no-book configurations."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
import re
import selectors
import shutil
import tempfile
import subprocess
import sys
from collections import defaultdict

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cshogi


DEFAULT_ENGINE = Path("/home/hmatsuya/workspace/Shogi/nagisa-bench/NAGISA_V4")
DEFAULT_PUCT_BOOK = Path("output/punct_book.yaneuraou.db")
DEFAULT_TERASHOCK_BOOK = Path("/home/hmatsuya/workspace/Shogi/data/terashock/new_petabook_20250505c.db")

# BookEval*Diff per staged book, same for both colours: the PUCT book gets 40 (varied lines), the Terashock book 0 (its top move is
# its strongest). A book-less engine never reads it.
BOOK_EVAL_DIFF = {"user_book1.db": "40", "user_book2.db": "0"}

# A book hit prints only "nodes 0" info lines; a real search ends with a non-zero count.
# The rule reads the last info line that carries a node count, ignoring "info string" and lines without one.
# shortcut: "nodes 0" marks any skipped search, not only book hits (declaration win and resign
# also skip). With the default EnteringKingRule a declaration win replies "win", which ends the
# game before a move is counted; TryRule would be miscounted. Upgrade if that rule is ever used.
_NODES_RE = re.compile(r"\bnodes (\d+)")


class BenchmarkError(RuntimeError):
    """Raised when a USI engine cannot complete a protocol operation."""


@dataclass(frozen=True)
class BookStats:
    entries: int
    records: int


def stage_books(puct_source: Path, terashock_source: Path, book_dir: Path) -> tuple[BookStats, BookStats]:
    """Place both text books under YaneuraOu's accepted BookFile names."""
    book_dir.mkdir(parents=True, exist_ok=True)
    puct_dest = book_dir / "user_book1.db"
    terashock_dest = book_dir / "user_book2.db"
    shutil.copy2(puct_source, puct_dest)
    shutil.copy2(terashock_source, terashock_dest)

    def stats(path: Path) -> BookStats:
        entries = records = 0
        with path.open(encoding="utf-8-sig") as f:   # stream: the petabook is 493 MB
            for line in f:
                if line.startswith("sfen "):
                    entries += 1
                elif line.strip() and not line.startswith("#"):
                    records += 1
        return BookStats(entries, records)

    return stats(puct_dest), stats(terashock_dest)


class USIProcess:
    def __init__(self, executable: Path, book_file: str, book_dir: Path, timeout: float, nodes: int):
        self.timeout = timeout
        self.has_book = book_file != "no_book"
        self.process = subprocess.Popen(
            [str(executable)],
            cwd=str(executable.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self._stdout_fd = self.process.stdout.fileno()
        self._buffer = b""
        self._selector = selectors.DefaultSelector()
        self._selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self._send("usi")
            self._wait_for("usiok")
            options = {
                "Threads": "4",
                "NodesLimit": str(nodes),
                "USI_Ponder": "false",
                "USI_OwnBook": "false" if book_file == "no_book" else "true",
                "BookFile": book_file,
                "BookDir": str(book_dir),
                "BookDepthWhiteLimit": "0",
                "BookMoves": "320",
                "BookOnTheFly": "true",
                "BookEvalBlackDiff": BOOK_EVAL_DIFF.get(book_file, "0"),
                "BookEvalWhiteDiff": BOOK_EVAL_DIFF.get(book_file, "0"),
                "BookEvalBlackLimit": "-500",
                "BookEvalWhiteLimit": "-500",
                "MaxMovesToDraw": "320",
                "ResignValue": "99999",
                "MinimumThinkingTime": "1",
                "NetworkDelay": "0",
                "NetworkDelay2": "0",
                "RoundUpToFullSecond": "false",
                "USI_Hash": "64",
            }
            for name, value in options.items():
                self._send(f"setoption name {name} value {value}")
            self._send("isready")
            while True:
                line = self._read_line()
                if "Error!" in line or "combo value not found" in line:
                    raise BenchmarkError(f"engine startup failed: {line}")
                if line == "readyok":
                    break

        except Exception:
            self.close()
            raise

    def _send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise BenchmarkError(f"engine exited before command: {command}")
        assert self.process.stdin is not None
        self.process.stdin.write((command + "\n").encode())
        self.process.stdin.flush()

    def _read_line(self) -> str:
        while b"\n" not in self._buffer:
            if not self._selector.select(self.timeout):
                raise BenchmarkError(f"engine read timed out after {self.timeout:g}s")
            chunk = os.read(self._stdout_fd, 65536)
            if not chunk:
                raise BenchmarkError(f"engine stdout closed (status {self.process.poll()})")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line.decode("utf-8", errors="replace").rstrip("\r")

    def _wait_for(self, expected: str) -> None:
        while self._read_line() != expected:
            pass

    def bestmove(self, moves: list[str], nodes: int) -> tuple[str, bool]:
        """Return (move, from_book). A move is from the book when the engine has one
        and the last info line that carries a node count, before ``bestmove``, reports ``nodes 0``."""
        position = "position startpos" + ((" moves " + " ".join(moves)) if moves else "")
        self._send(position)
        self._send(f"go nodes {nodes}")
        last_nodes = None
        while True:
            line = self._read_line()
            if line.startswith("info ") and not line.startswith("info string"):
                found = _NODES_RE.search(line)
                if found:
                    last_nodes = int(found.group(1))
            elif line.startswith("bestmove "):
                fields = line.split()
                if len(fields) < 2:
                    raise BenchmarkError(f"malformed bestmove: {line!r}")
                return fields[1], self.has_book and last_nodes == 0

    def gameover(self, result: str) -> None:
        self._send(f"gameover {result}")

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired, BenchmarkError):
                self.process.kill()
                self.process.wait()
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self._selector.close()


def engine_for_color(color: int, a_is_black: bool) -> int:
    return 0 if color == (cshogi.BLACK if a_is_black else cshogi.WHITE) else 1


@dataclass
class GameRecord:
    """One finished game. ``result`` is a_win, b_win, draw or aborted."""
    result: str
    a_is_black: bool
    a_book_moves: int = 0
    a_moves: int = 0
    b_book_moves: int = 0
    b_moves: int = 0


def finish(engine_a: USIProcess, engine_b: USIProcess, record: GameRecord, result: str) -> GameRecord:
    engine_a.gameover("win" if result == "a_win" else "lose" if result == "b_win" else "draw")
    engine_b.gameover("win" if result == "b_win" else "lose" if result == "a_win" else "draw")
    record.result = result
    return record


def play_game(engine_a: USIProcess, engine_b: USIProcess, nodes: int, a_is_black: bool, max_ply: int) -> GameRecord:
    board = cshogi.Board()
    moves: list[str] = []
    record = GameRecord("aborted", a_is_black)
    repetitions = defaultdict(int)
    repetitions[board.zobrist_hash()] = 1
    engine_a._send("usinewgame")
    engine_b._send("usinewgame")
    try:
        for ply in range(max_ply):
            repetition = board.is_draw()
            if repetitions[board.zobrist_hash()] < 4:
                repetition = cshogi.NOT_REPETITION
            if repetition in (cshogi.REPETITION_WIN, cshogi.REPETITION_LOSE):
                winner = board.turn if repetition == cshogi.REPETITION_WIN else 1 - board.turn
                return finish(engine_a, engine_b, record, "a_win" if engine_for_color(winner, a_is_black) == 0 else "b_win")
            if repetition in (cshogi.REPETITION_DRAW, cshogi.REPETITION_SUPERIOR, cshogi.REPETITION_INFERIOR):
                return finish(engine_a, engine_b, record, "draw")
            index = engine_for_color(board.turn, a_is_black)
            move_text, from_book = (engine_a if index == 0 else engine_b).bestmove(moves, nodes)
            # resign / win / 0000 end the game and are not moves, so they are not counted as moves or book moves.
            if move_text in ("resign", "win") or move_text == "0000":
                if move_text == "win":
                    return finish(engine_a, engine_b, record, "a_win" if index == 0 else "b_win")
                return finish(engine_a, engine_b, record, "b_win" if index == 0 else "a_win")
            move = board.move_from_usi(move_text)
            if move == cshogi.MOVE_NONE or move not in board.legal_moves:
                raise BenchmarkError(f"illegal bestmove {move_text!r} at ply {ply}")
            if index == 0:
                record.a_moves += 1
                record.a_book_moves += from_book
            else:
                record.b_moves += 1
                record.b_book_moves += from_book
            board.push(move)
            repetitions[board.zobrist_hash()] += 1
            moves.append(move_text)
        return finish(engine_a, engine_b, record, "draw")
    except Exception as exc:
        print(f"benchmark game aborted: {exc!r}", file=sys.stderr)
        return record  # still "aborted": finish() is the only place a result is set


def _rate(wins: int, games: int) -> Optional[float]:
    return wins / games if games else None


def summarize(name_a: str, name_b: str, records: list[GameRecord], games_requested: int) -> dict[str, object]:
    """Turn finished games into the pairing report. Aborted games are counted but excluded
    from every rate, so every rate uses completed games only. ``book_share`` is the exception
    in its denominator: it divides by moves, not games."""
    done = [r for r in records if r.result != "aborted"]

    def side(a_black: bool) -> dict[str, object]:
        # Games in which engine A played Black (a_black) or White.
        g = [r for r in done if r.a_is_black == a_black]
        a_w = sum(r.result == "a_win" for r in g)
        b_w = sum(r.result == "b_win" for r in g)
        d = len(g) - a_w - b_w
        return {"games": len(g), "a_wins": a_w, "b_wins": b_w, "draws": d,
                "a_score": _rate(a_w + 0.5 * d, len(g))}

    as_black, as_white = side(True), side(False)
    wins_a = as_black["a_wins"] + as_white["a_wins"]
    wins_b = as_black["b_wins"] + as_white["b_wins"]
    draws = as_black["draws"] + as_white["draws"]
    # Black side wins: A's wins while A was Black plus B's wins while A was White.
    black_wins = as_black["a_wins"] + as_white["b_wins"]
    white_wins = as_black["b_wins"] + as_white["a_wins"]

    def book(used: int, total: int) -> dict[str, object]:
        return {"book_moves": used, "moves": total, "book_share": _rate(used, total),
                "book_moves_per_game": _rate(used, len(done))}

    return {
        "a": name_a, "b": name_b, "games_requested": games_requested, "completed": len(done),
        "a_wins": wins_a, "b_wins": wins_b, "draws": draws, "aborted": len(records) - len(done),
        "score_a": _rate(wins_a + 0.5 * draws, len(done)),
        "black_wins": black_wins, "white_wins": white_wins,
        "black_win_rate": _rate(black_wins, len(done)), "white_win_rate": _rate(white_wins, len(done)),
        "a_as_black": as_black, "a_as_white": as_white,
        "book_a": book(sum(r.a_book_moves for r in done), sum(r.a_moves for r in done)),
        "book_b": book(sum(r.b_book_moves for r in done), sum(r.b_moves for r in done)),
    }


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def format_pair(pair: dict[str, object]) -> str:
    """Human-readable report for one pairing (the JSON keeps the full detail)."""
    a, b = pair["a"], pair["b"]
    lines = [f"{a} vs {b}: {pair['a_wins']}-{pair['b_wins']}-{pair['draws']} (aborted {pair['aborted']}), "
             f"score_{a}={_pct(pair['score_a'])}",
             f"  side win rate: black {_pct(pair['black_win_rate'])} ({pair['black_wins']}), "
             f"white {_pct(pair['white_win_rate'])} ({pair['white_wins']})"]
    for key, label in (("a_as_black", f"{a} as black"), ("a_as_white", f"{a} as white")):
        s = pair[key]
        lines.append(f"  {label}: {s['a_wins']}-{s['b_wins']}-{s['draws']} of {s['games']}, score {_pct(s['a_score'])}")
    for key, name in (("book_a", a), ("book_b", b)):
        s = pair[key]
        if s["moves"] == 0:
            lines.append(f"  book moves {name}: n/a (no completed games)")
        else:
            lines.append(f"  book moves {name}: {s['book_moves']} of {s['moves']} ({_pct(s['book_share'])}), "
                         f"{s['book_moves_per_game']:.1f}/game")
    return "\n".join(lines)


def run_pair(name_a: str, file_a: str, name_b: str, file_b: str, *, engine: Path, book_dir: Path, games: int, nodes: int, timeout: float, max_ply: int, concurrency: int) -> dict[str, object]:
    records: list[GameRecord] = []

    def start():
        return (USIProcess(engine, file_a, book_dir, timeout, nodes), USIProcess(engine, file_b, book_dir, timeout, nodes))

    # Each lane owns two engine processes. Games in a pairing run concurrently.
    lanes = [start() for _ in range(min(concurrency, games))]
    pending = list(range(games))
    executor = ThreadPoolExecutor(max_workers=len(lanes))
    try:
        futures = {}
        for lane, (a, b) in enumerate(lanes):
            if pending:
                game = pending.pop(0)
                futures[executor.submit(play_game, a, b, nodes, game % 2 == 0, max_ply)] = (lane, game, a, b)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                lane, game, a, b = futures.pop(future)
                records.append(future.result())
                a.close()
                b.close()
                if pending:
                    a, b = start()
                    lanes[lane] = (a, b)
                    next_game = pending.pop(0)
                    futures[executor.submit(play_game, a, b, nodes, next_game % 2 == 0, max_ply)] = (lane, next_game, a, b)
    finally:
        executor.shutdown(wait=True)
        for a, b in lanes:
            a.close()
            b.close()
    return summarize(name_a, name_b, records, games)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--book-dir", type=Path, default=None)
    parser.add_argument("--puct-book", type=Path, default=DEFAULT_PUCT_BOOK)
    parser.add_argument("--terashock-book", type=Path, default=DEFAULT_TERASHOCK_BOOK)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--max-ply", type=int, default=320)
    parser.add_argument("--nodes", type=int, default=100000)
    parser.add_argument("--concurrency", type=int, default=8, help="global concurrent games across all pairings")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("output/book-benchmark.json"))
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args(argv)
    if args.games < 1 or args.concurrency < 1:
        raise SystemExit("games and concurrency must be positive")
    args.engine = args.engine.absolute()
    if args.book_dir is not None:
        args.book_dir = args.book_dir.resolve()
    else:
        args.book_dir = Path(tempfile.mkdtemp(prefix="yaneuraou-book-", dir="/tmp"))
    owns_book_dir = args.book_dir.name.startswith("yaneuraou-book-") and args.book_dir.parent == Path("/tmp")
    args.puct_book = args.puct_book.resolve(); args.terashock_book = args.terashock_book.resolve()
    for path in (args.engine, args.puct_book, args.terashock_book):
        if not path.is_file():
            raise SystemExit(f"missing file: {path}")
    puct_stats, terashock_stats = stage_books(args.puct_book, args.terashock_book, args.book_dir)
    pair_specs = [
        ("puct", "user_book1.db", "none", "no_book"),
        ("puct", "user_book1.db", "terashock", "user_book2.db"),
        ("terashock", "user_book2.db", "none", "no_book"),
    ]
    base, remainder = divmod(args.concurrency, len(pair_specs))
    pair_concurrencies = [base + (i < remainder) for i in range(len(pair_specs))]
    pair_concurrencies = [max(1, value) for value in pair_concurrencies]
    def run(item):
        spec, pair_concurrency = item
        a, file_a, b, file_b = spec
        return run_pair(a, file_a, b, file_b, engine=args.engine, book_dir=args.book_dir, games=args.games, nodes=args.nodes, timeout=args.timeout, max_ply=args.max_ply, concurrency=pair_concurrency)

    with ThreadPoolExecutor(max_workers=len(pair_specs)) as executor:
        pairs = list(executor.map(run, zip(pair_specs, pair_concurrencies)))




    result = {"metadata": {"engine": str(args.engine), "book_dir": str(args.book_dir), "puct_book": str(args.puct_book), "terashock_book": str(args.terashock_book), "games_per_pair": args.games, "nodes_per_move": args.nodes, "max_ply": args.max_ply, "concurrency": args.concurrency, "pair_concurrency": pair_concurrencies, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, "book_staging": {"puct": puct_stats.__dict__, "terashock": terashock_stats.__dict__}, "pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for pair in pairs:
        print(format_pair(pair))
    print(f"results: {args.output}")
    if args.keep_temp or not owns_book_dir:
        print(f"book directory: {args.book_dir}")
    else:
        if owns_book_dir:
            shutil.rmtree(args.book_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
