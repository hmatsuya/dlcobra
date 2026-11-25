"""Unit tests for the book benchmark's per-color and book-move accounting.

They exercise ``summarize`` and the ``nodes 0`` book-hit rule without starting an engine.
"""

from __future__ import annotations

import threading
import time

from dlshogi.utils.book_benchmark import GameRecord, format_pair, summarize


def game(result, a_black, a_book=0, a_moves=0, b_book=0, b_moves=0):
    return GameRecord(result, a_black, a_book_moves=a_book, a_moves=a_moves,
                      b_book_moves=b_book, b_moves=b_moves)


def test_black_and_white_win_rates_are_split_by_side_not_by_engine():
    # A is black in games 1-3, white in games 4-5.
    records = [
        game("a_win", True),    # A (black) wins -> black win
        game("a_win", True),    # A (black) wins -> black win
        game("b_win", True),    # B (white) wins -> white win
        game("b_win", False),   # B (black) wins -> black win
        game("draw", False),
    ]
    s = summarize("A", "B", records, 5)
    assert (s["black_wins"], s["white_wins"]) == (3, 1)
    assert s["black_win_rate"] == 0.6 and s["white_win_rate"] == 0.2
    assert (s["a_as_black"]["games"], s["a_as_white"]["games"]) == (3, 2)
    assert s["a_as_black"]["a_wins"] == 2 and s["a_as_black"]["b_wins"] == 1
    # The draw counts half: A scores 2 + 0.5 over 5 games, and as White (1 loss, 1 draw) 0.5 / 2.
    assert s["score_a"] == 2.5 / 5 and s["a_as_white"]["a_score"] == 0.25
    text = format_pair(s)
    assert "A as black: 2-1-0 of 3" in text and "A as white: 0-1-1 of 2" in text
    assert "black 60.0% (3), white 20.0% (1)" in text
    assert s["a_as_white"]["a_wins"] == 0 and s["a_as_white"]["draws"] == 1


def test_aborted_games_are_excluded_from_every_rate():
    records = [game("a_win", True), game("aborted", False), game("aborted", True)]
    s = summarize("A", "B", records, 3)
    assert s["completed"] == 1 and s["aborted"] == 2
    assert s["black_win_rate"] == 1.0 and s["white_win_rate"] == 0.0
    assert s["a_as_white"]["games"] == 0 and s["a_as_white"]["a_score"] is None


def test_book_moves_count_only_completed_games():
    records = [game("a_win", True, a_book=3, a_moves=10, b_book=0, b_moves=10),
               game("aborted", False, a_book=9, a_moves=9)]
    s = summarize("A", "B", records, 2)
    assert s["book_a"]["book_moves"] == 3 and s["book_a"]["moves"] == 10
    assert s["book_a"]["book_share"] == 0.3
    assert s["book_b"]["book_moves"] == 0 and s["book_b"]["book_share"] == 0.0


def test_pairing_with_no_completed_games_does_not_divide_by_zero():
    s = summarize("A", "B", [game("aborted", True)], 1)
    assert s["black_win_rate"] is None and s["book_a"]["book_share"] is None
    assert "n/a" in format_pair(s)


def _engine_reply(has_book, info_lines, bestmove="bestmove 7g7f ponder 3c3d"):
    """Run USIProcess.bestmove against a scripted engine output, with no process."""
    from dlshogi.utils.book_benchmark import USIProcess

    engine = USIProcess.__new__(USIProcess)  # skip __init__: no engine is started
    engine.has_book = has_book
    engine._send = lambda command: None
    replies = iter(info_lines + [bestmove])
    engine._read_line = lambda: next(replies)
    return engine.bestmove([], 1000)


def test_book_hit_needs_nodes_zero_and_an_engine_with_a_book():
    hit = ["info depth 0 multipv 1 score cp 80 nodes 0 nps 0 time 3 pv 7g7f"]
    searched = ["info depth 9 score cp 147 nodes 20052 nps 3342000 pv 7g7f"]
    assert _engine_reply(True, hit) == ("7g7f", True)
    assert _engine_reply(True, searched) == ("7g7f", False)
    # An engine with no book never reports a book move, even if it prints nodes 0.
    assert _engine_reply(False, hit) == ("7g7f", False)


def test_book_hit_uses_the_last_info_line_and_ignores_info_string():
    # The info-string line carries a non-zero node count: only the "info string" guard keeps it from deciding the answer.
    mixed = ["info depth 0 score cp 43 nodes 0 nps 0 pv 2g2f",
             "info string cache report nodes 900",
             "info depth 0 score cp 80 nodes 0 nps 0 pv 7g7f"]
    assert _engine_reply(True, mixed)[1] is True
    # An "info string" line that happens to say "nodes 0" must not make a searched move look like a book hit.
    assert _engine_reply(True, ["info depth 5 nodes 900 pv 7g7f", "info string cleared nodes 0 entries"])[1] is False
    # A search that starts with a zero-node line but ends with real nodes is not a book hit.
    assert _engine_reply(True, ["info depth 0 nodes 0 pv 7g7f", "info depth 5 nodes 900 pv 7g7f"])[1] is False


class _Scripted:
    """Stands in for USIProcess: replays (move, from_book) answers and records gameover calls."""

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def new_game(self):
        self.sent.append("new_game")

    def bestmove(self, moves, nodes):
        return self.replies.pop(0)

    def gameover(self, result):
        self.sent.append("gameover " + result)


def test_a_declaration_win_ends_the_game_without_counting_a_book_move():
    from dlshogi.utils.book_benchmark import play_game

    a, b = _Scripted([("win", True)]), _Scripted([])
    record = play_game(a, b, 1000, True, 320)
    assert record.result == "a_win" and record.a_moves == 0 and record.a_book_moves == 0
    assert a.sent[0] == b.sent[0] == "new_game"  # each game resets both reused engines first


def test_play_game_attributes_moves_and_colours_to_the_right_engine():
    from dlshogi.utils.book_benchmark import play_game

    # A is White, so B (Black) moves first. B plays 7g7f, 2g2f (both book); A plays 3c3d (book), 8c8d (search).
    # Different book counts per engine (A=1, B=2) so a mix-up between the two cannot cancel out.
    a = _Scripted([("3c3d", True), ("8c8d", False)])
    b = _Scripted([("7g7f", True), ("2g2f", True), ("resign", False)])
    record = play_game(a, b, 1000, False, 320)
    assert record.a_is_black is False and record.result == "a_win"
    assert (record.a_moves, record.a_book_moves) == (2, 1)
    assert (record.b_moves, record.b_book_moves) == (2, 2)
    # B resigned, so A is told it won and B is told it lost.
    assert "gameover win" in a.sent and "gameover lose" in b.sent


# Captured from the real YaneuraOu build (V2 options, PUCT book): one book hit, one search that left the book, and one no-book search.
REAL = {
 "book_hit": [
  "info depth 0 multipv 1 score cp 43 nodes 0 nps 0 hashfull 0 time 0 pv 2g2f 4a3b 2f2e 8c8d 6i7h 8d8e 9g9f 5a5b (71.21%)",
  "info depth 0 multipv 2 score cp 80 nodes 0 nps 0 hashfull 0 time 0 pv 7g7f 8c8d 2g2f 8d8e 2f2e 4a3b 8h7g 3c3d (28.49%)",
  "info depth 0 multipv 3 score cp 36 nodes 0 nps 0 hashfull 0 time 0 pv 6i7h 3c3d 7g7f 8c8d 2g2f 4c4d 7i6h 4a3b (0.29%)",
  "info depth 0 multipv 4 score cp 0 nodes 0 nps 0 hashfull 0 time 0 pv 9g9f 8c8d (0.01%)",
  "info depth 0 multipv 5 score cp -25 nodes 0 nps 0 hashfull 0 time 0 pv 6g6f 3c3d 7g7f 8c8d 1g1f 8d8e 8h7g 7a6b (0.00%)",
  "info depth 0 multipv 6 score cp -10 nodes 0 nps 0 hashfull 0 time 0 pv 1g1f (0.00%)",
  "info string BookEvalBlackDiff = 120 , BookEvalBlackLimit = 0 , 6 moves to 4 moves.",
  "info depth 0 multipv 1 score cp 36 nodes 0 nps 0 hashfull 0 time 3 pv 6i7h 8c8d",
  "bestmove 6i7h ponder 8c8d"
 ],
 "book_miss_searched": [
  "info depth 3 seldepth 6 multipv 1 score cp 163 upperbound nodes 2011 nps 1005500 hashfull 0 time 2 pv 2e2d 2c2d",
  "bestmove 2e2d ponder 2c2d"
 ],
 "no_book_search": [
  "info depth 3 seldepth 8 multipv 1 score cp 206 upperbound nodes 2024 nps 1012000 hashfull 0 time 2 pv 2f2e 4a3b",
  "bestmove 2f2e ponder 4a3b"
 ]
}


def test_real_transcripts_book_hit_is_detected_and_searches_are_not():
    def reply(has_book, lines):
        # The captured transcript ends with the engine's own "bestmove" line; replay it unchanged.
        return _engine_reply(has_book, lines[:-1], lines[-1])

    assert reply(True, REAL["book_hit"]) == ("6i7h", True)
    assert reply(True, REAL["book_miss_searched"]) == ("2e2d", False)
    assert reply(False, REAL["no_book_search"]) == ("2f2e", False)


def test_silent_zero_inputs_are_not_book_hits():
    # No info line at all, or info lines with no node count: never a book hit.
    assert _engine_reply(True, []) == ("7g7f", False)
    assert _engine_reply(True, ["info depth 1 score cp 10 pv 7g7f"]) == ("7g7f", False)
    # The rule uses the last info line that carries a node count.
    assert _engine_reply(True, ["info depth 0 nodes 0 pv 7g7f", "info depth 1 pv 7g7f"]) == ("7g7f", True)



def test_book_eval_diff_depends_on_book_not_colour(tmp_path):
    import sys
    from dlshogi.utils.book_benchmark import USIProcess

    # A fake engine that answers the handshake and records every setoption line.
    log = tmp_path / "options.log"
    engine = tmp_path / "engine.py"
    engine.write_text(
        "#!%s\nimport sys\nlog = open(%r, 'a')\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if line.startswith('setoption'): log.write(line + '\\n'); log.flush()\n"
        "    if line == 'usi': print('usiok', flush=True)\n"
        "    if line == 'isready': print('readyok', flush=True)\n"
        "    if line == 'quit': break\n" % (sys.executable, str(log))
    )
    engine.chmod(0o755)
    for book, expected in (("user_book1.db", "40"), ("user_book2.db", "0"), ("no_book", "0")):
        log.write_text("")
        USIProcess(engine, book, tmp_path, 5.0, 1000).close()
        sent = {l.split()[2]: l.split()[4] for l in log.read_text().splitlines()}
        assert sent["BookEvalBlackDiff"] == sent["BookEvalWhiteDiff"] == expected, book


def _fake_engine(tmp_path):
    import sys

    log = tmp_path / "commands.log"
    script = tmp_path / "engine.py"
    script.write_text(
        "#!%s\nimport sys\nlog = open(%r, 'a')\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    log.write(line + '\\n'); log.flush()\n"
        "    if line == 'usi': print('usiok', flush=True)\n"
        "    if line == 'isready': print('readyok', flush=True)\n"
        "    if line == 'quit': break\n" % (sys.executable, str(log))
    )
    script.chmod(0o755)
    return script, log


def test_new_game_waits_for_readyok_and_rejects_error(tmp_path):
    import sys

    import pytest
    from dlshogi.utils.book_benchmark import BenchmarkError, USIProcess

    engine, log = _fake_engine(tmp_path)
    process = USIProcess(engine, "no_book", tmp_path, 5.0, 1000)
    log.write_text("")
    process.new_game()
    assert log.read_text().split() == ["usinewgame", "isready"]
    process.close()

    # new_game() must read through its own readyok: the engine answers the reset's isready with a progress line first,
    # so a new_game() that reads nothing, or stops after one line, leaves a stale line in front of the next reply.
    slow = tmp_path / "slow.py"
    slow.write_text(
        "#!%s\nimport sys\nn = 0\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if line == 'usi': print('usiok', flush=True)\n"
        "    if line == 'isready':\n"
        "        n += 1\n"
        "        if n > 1: print('info string loading', flush=True)\n"
        "        print('readyok', flush=True)\n"
        "    if line == 'quit': break\n" % sys.executable
    )
    slow.chmod(0o755)
    process = USIProcess(slow, "no_book", tmp_path, 5.0, 1000)
    process.new_game()
    process._send("usi")
    assert process._read_line() == "usiok"
    process.close()

    # an engine that answers the startup isready, then reports an error on the next one
    script = tmp_path / "err.py"
    script.write_text(
        "#!%s\nimport sys\nn = 0\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if line == 'usi': print('usiok', flush=True)\n"
        "    if line == 'isready':\n"
        "        n += 1\n"
        "        print('readyok' if n == 1 else 'info string Error! : boom', flush=True)\n"
        "        if n > 1: print('readyok', flush=True)\n"
        "    if line == 'quit': break\n" % sys.executable
    )
    script.chmod(0o755)
    process = USIProcess(script, "no_book", tmp_path, 5.0, 1000)
    with pytest.raises(BenchmarkError, match="new game failed"):
        process.new_game()
    process.close()


SPECS = [
    ("puct", "user_book1.db", "none", "no_book"),
    ("puct", "user_book1.db", "terashock", "user_book2.db"),
    ("terashock", "user_book2.db", "none", "no_book"),
]


def _run_pairs_with_stubs(monkeypatch, tmp_path, *, games, concurrency, abort_on=(), fail_start_on=None, barrier_first=0, hold_until_other_raises=False, park_until_shutdown=False, closed_in_use=None, raise_on=()):
    """Run run_pairs with stub engines. Returns (started ids, closed ids, played, results, error).

    ``played`` holds (a book, b book, a_is_black, a id, b id) per game in call order; ``abort_on`` are 1-based call numbers
    (deterministic only with concurrency 1); the first ``barrier_first`` games wait for each other, so they only pass if that
    many games really run at the same time; with ``hold_until_other_raises`` the PUCT-vs-Terashock game stays in flight until the
    PUCT-vs-no-book game raises, and ``closed_in_use`` collects engines closed while a game was still using them; with
    ``park_until_shutdown`` the second game waits until run_pairs has cancelled its queue; the games in ``raise_on`` raise a
    BenchmarkError instead of returning."""
    from dlshogi.utils import book_benchmark as bb

    started, closed, played = [], [], []
    lock = threading.Lock()
    barrier = threading.Barrier(barrier_first, timeout=5) if barrier_first else None
    in_flight, raised, shutting_down = threading.Event(), threading.Event(), threading.Event()
    closed_in_use = [] if closed_in_use is None else closed_in_use

    class Stub:
        def __init__(self, engine, book_file, book_dir, timeout, nodes):
            with lock:
                if fail_start_on is not None and len(started) == fail_start_on:
                    raise bb.BenchmarkError("start failed")
                self.id = len(started)
                started.append(self.id)
            self.book = book_file

        def close(self):
            with lock:
                if self.id in in_use:
                    closed_in_use.append(self.id)
                closed.append(self.id)

    calls = {"n": 0}

    in_use: set = set()

    def fake_play_game(a, b, nodes, a_is_black, max_ply):
        with lock:
            calls["n"] += 1
            n = calls["n"]
            assert a.id not in closed and b.id not in closed  # a closed (poisoned) engine never gets a game
            assert not ({a.id, b.id} & in_use) and a.id != b.id  # no engine is in two games at once
            in_use.update((a.id, b.id))
            played.append((a.book, b.book, a_is_black, a.id, b.id))
        try:
            if barrier is not None and n <= barrier_first:
                barrier.wait()
            if hold_until_other_raises and (a.book, b.book) == ("user_book1.db", "user_book2.db"):
                in_flight.set()
                raised.wait(5)
                time.sleep(0.2)  # the window in which a premature close_all would show
            elif hold_until_other_raises and (a.book, b.book) == ("user_book1.db", "no_book"):
                in_flight.wait(5)
                raised.set()
                raise bb.BenchmarkError("game failed")
            if park_until_shutdown and n == 2:
                shutting_down.wait(5)
            if n in raise_on:
                raise bb.BenchmarkError("game failed")
            return bb.GameRecord("aborted" if n in abort_on else "a_win", a_is_black)
        finally:
            with lock:
                in_use.difference_update((a.id, b.id))

    class ParkingExecutor(bb.ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            super().shutdown(wait=False, cancel_futures=cancel_futures)  # cancel the queue first,
            shutting_down.set()  # then let the parked game go
            super().shutdown(wait=wait)

    monkeypatch.setattr(bb, "USIProcess", Stub)
    monkeypatch.setattr(bb, "play_game", fake_play_game)
    if park_until_shutdown:
        monkeypatch.setattr(bb, "ThreadPoolExecutor", ParkingExecutor)
    try:
        results = bb.run_pairs(SPECS, engine=tmp_path, book_dir=tmp_path, games=games, nodes=1, timeout=1.0, max_ply=1, concurrency=concurrency)
        error = None
    except bb.BenchmarkError as exc:
        results, error = None, str(exc)
    return started, sorted(closed), played, results, error


def test_run_pairs_shares_engines_across_pairings_and_alternates_colours(monkeypatch, tmp_path):
    started, closed, played, results, _ = _run_pairs_with_stubs(monkeypatch, tmp_path, games=4, concurrency=1)
    # one worker: three engines in total (one per book), not one pair per game or per pairing
    assert len(started) == 3 and set(closed) == {0, 1, 2}
    assert [r["completed"] for r in results] == [4, 4, 4]
    for book_a, book_b in (("user_book1.db", "no_book"), ("user_book1.db", "user_book2.db"), ("user_book2.db", "no_book")):
        assert [c for a, b, c, *_ in played if (a, b) == (book_a, book_b)] == [True, False, True, False]


def test_run_pairs_keeps_every_worker_busy_across_pairings(monkeypatch, tmp_path):
    # 6 games must run at once, though a pairing has only 2 games: the workers are shared, not split per pairing
    _, _, _, results, error = _run_pairs_with_stubs(monkeypatch, tmp_path, games=2, concurrency=6, barrier_first=6)
    assert error is None and [r["completed"] for r in results] == [2, 2, 2]


def test_run_pairs_replaces_only_the_engines_of_an_aborted_game(monkeypatch, tmp_path):
    # call 2 is the first PUCT-vs-Terashock game: both of its engines are closed, the no-book engine is untouched
    started, closed, played, results, _ = _run_pairs_with_stubs(monkeypatch, tmp_path, games=2, concurrency=1, abort_on=(2,))
    assert results[1]["aborted"] == 1 and results[0]["aborted"] == 0 and results[2]["aborted"] == 0
    assert len(started) == 5 and set(closed) == {0, 1, 2, 3, 4}  # 3 books + 2 replacements; every engine closed
    assert played[1][3] not in played[3][3:]


def test_run_pairs_closes_every_started_engine_when_start_fails(monkeypatch, tmp_path):
    # the second engine of the first game cannot start: the first one must still be closed
    started, closed, _, _, error = _run_pairs_with_stubs(monkeypatch, tmp_path, games=2, concurrency=1, fail_start_on=1)
    assert error == "start failed" and started == [0] and set(closed) == {0}


def test_run_pairs_reuses_pooled_engines_between_concurrent_games(monkeypatch, tmp_path):
    # 3 workers, 8 games per pairing: a pairing's two books differ, so a worker holds at most one engine per book,
    # and 3 workers never need more than 3 x 3 = 9 engines in total, whatever the timing
    started, closed, played, results, _ = _run_pairs_with_stubs(monkeypatch, tmp_path, games=8, concurrency=3)
    assert [r["completed"] for r in results] == [8, 8, 8] and len(played) == 24
    assert len(started) <= 9 and set(closed) == set(started)  # the in_use assertion in the helper guards sharing


def test_run_pairs_waits_for_games_in_flight_before_closing_engines_on_failure(monkeypatch, tmp_path):
    # the PUCT-vs-no-book game (first in the queue) fails while the PUCT-vs-Terashock game is still playing:
    # run_pairs raises at once, but must not close that game's engines until it has finished
    in_use = []
    started, closed, _, _, error = _run_pairs_with_stubs(monkeypatch, tmp_path, games=2, concurrency=2, hold_until_other_raises=True, closed_in_use=in_use)
    assert error == "game failed"
    assert in_use == []  # no engine was closed while a game used it
    assert set(closed) == set(started)  # and every engine that started is closed afterwards


def test_run_pairs_does_not_start_queued_games_after_a_failure(monkeypatch, tmp_path):
    # one worker: game 1 fails. The worker may already have taken game 2, and parks there until run_pairs has cancelled the
    # queue, so game 3 onwards can never start, whatever the thread timing (without the cancel all 9 games would run)
    started, closed, played, _, error = _run_pairs_with_stubs(monkeypatch, tmp_path, games=3, concurrency=1, raise_on=(1,), park_until_shutdown=True)
    assert error == "game failed" and 1 <= len(played) <= 2 and set(closed) == set(started)


def test_a_hung_reset_aborts_the_game_instead_of_the_run():
    from dlshogi.utils.book_benchmark import BenchmarkError, play_game

    class Hangs:
        def new_game(self):
            raise BenchmarkError("engine read timed out after 1s")

    class Fine:
        def new_game(self):
            pass

    for a, b in ((Hangs(), Fine()), (Fine(), Hangs())):
        record = play_game(a, b, 1000, True, 320)
        assert record.result == "aborted"
