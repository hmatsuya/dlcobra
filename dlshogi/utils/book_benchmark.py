"""Pairwise CPU benchmark for the PUCT, Terashock, and no-book configurations."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
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


DEFAULT_ENGINE = Path("/home/hmatsuya/workspace/Shogi/YaneuraOuOriginal/exe/YaneuraOu-by-gcc")
DEFAULT_PUCT_BOOK = Path("output/punct_book.yaneuraou.db")
DEFAULT_TERASHOCK_BOOK = Path("/home/hmatsuya/workspace/Shogi/data/terashock/user_book1.db")


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
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("sfen "):
                entries += 1
            elif line and not line.startswith("#"):
                records += 1
        return BookStats(entries, records)

    return stats(puct_dest), stats(terashock_dest)


class USIProcess:
    def __init__(self, executable: Path, book_file: str, book_dir: Path, timeout: float, nodes: int):
        self.timeout = timeout
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
                "BookEvalBlackDiff": "120",
                "BookEvalWhiteDiff": "120",
                "BookEvalWhiteLimit": "-150",
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

    def bestmove(self, moves: list[str], nodes: int) -> str:
        position = "position startpos" + ((" moves " + " ".join(moves)) if moves else "")
        self._send(position)
        self._send(f"go nodes {nodes}")
        while True:
            line = self._read_line()
            if line.startswith("bestmove "):
                fields = line.split()
                if len(fields) < 2:
                    raise BenchmarkError(f"malformed bestmove: {line!r}")
                return fields[1]

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


def finish(engine_a: USIProcess, engine_b: USIProcess, result: str) -> str:
    engine_a.gameover("win" if result == "a_win" else "lose" if result == "b_win" else "draw")
    engine_b.gameover("win" if result == "b_win" else "lose" if result == "a_win" else "draw")
    return result


def play_game(engine_a: USIProcess, engine_b: USIProcess, nodes: int, a_is_black: bool, max_ply: int) -> str:
    board = cshogi.Board()
    moves: list[str] = []
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
                return finish(engine_a, engine_b, "a_win" if engine_for_color(winner, a_is_black) == 0 else "b_win")
            if repetition in (cshogi.REPETITION_DRAW, cshogi.REPETITION_SUPERIOR, cshogi.REPETITION_INFERIOR):
                return finish(engine_a, engine_b, "draw")
            index = engine_for_color(board.turn, a_is_black)
            move_text = (engine_a if index == 0 else engine_b).bestmove(moves, nodes)
            if move_text in ("resign", "win") or move_text == "0000":
                if move_text == "win":
                    return finish(engine_a, engine_b, "a_win" if index == 0 else "b_win")
                return finish(engine_a, engine_b, "b_win" if index == 0 else "a_win")
            move = board.move_from_usi(move_text)
            if move == cshogi.MOVE_NONE or move not in board.legal_moves:
                raise BenchmarkError(f"illegal bestmove {move_text!r} at ply {ply}")
            board.push(move)
            repetitions[board.zobrist_hash()] += 1
            moves.append(move_text)
        return finish(engine_a, engine_b, "draw")
    except Exception as exc:
        print(f"benchmark game aborted: {exc!r}", file=sys.stderr)
        return "aborted"


def run_pair(name_a: str, file_a: str, name_b: str, file_b: str, *, engine: Path, book_dir: Path, games: int, nodes: int, timeout: float, max_ply: int, concurrency: int) -> dict[str, object]:
    counts = {"a_wins": 0, "b_wins": 0, "draws": 0, "aborted": 0}

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
                result = future.result()
                if result == "a_win":
                    counts["a_wins"] += 1
                elif result == "b_win":
                    counts["b_wins"] += 1
                elif result == "draw":
                    counts["draws"] += 1
                else:
                    counts["aborted"] += 1
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

    completed = counts["a_wins"] + counts["b_wins"] + counts["draws"]
    counts.update({"a": name_a, "b": name_b, "games_requested": games, "completed": completed, "score_a": (counts["a_wins"] + 0.5 * counts["draws"]) / completed if completed else None})
    return counts


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
        print(f"{pair['a']} vs {pair['b']}: {pair['a_wins']}-{pair['b_wins']}-{pair['draws']} (aborted {pair['aborted']}), score_a={pair['score_a']}")
    print(f"results: {args.output}")
    if args.keep_temp or not owns_book_dir:
        print(f"book directory: {args.book_dir}")
    else:
        if owns_book_dir:
            shutil.rmtree(args.book_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
