"""Unit tests for the book benchmark's per-color and book-move accounting.

They exercise ``summarize`` and the ``nodes 0`` book-hit rule without starting an engine.
"""

from __future__ import annotations

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

    def _send(self, command):
        self.sent.append(command)

    def bestmove(self, moves, nodes):
        return self.replies.pop(0)

    def gameover(self, result):
        self.sent.append("gameover " + result)


def test_a_declaration_win_ends_the_game_without_counting_a_book_move():
    from dlshogi.utils.book_benchmark import play_game

    a, b = _Scripted([("win", True)]), _Scripted([])
    record = play_game(a, b, 1000, True, 320)
    assert record.result == "a_win" and record.a_moves == 0 and record.a_book_moves == 0


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
