import unittest
from dlshogi.utils.plaintext_to_hcpe import flip_sfen
from dlshogi.utils.plaintext_to_hcpe import flip_sfen_square, flip_sfen_move

class TestFlipSfen(unittest.TestCase):
    def test_flip_sfen_basic(self):
        sfen = "lnsgk2nl/1r4gs1/p1pppp1pp/1p4p2/7P1/2P6/PP1PPPP1P/1SG4R1/LN2KGSNL b Bb 1"
        expected_flipped = "ln2kgsnl/1sg4r1/pp1pppp1p/2p4p1/1P7/6P2/P1PPPP1PP/1R4GS1/LNSGK2NL b Bb 1"
        self.assertEqual(flip_sfen(sfen), expected_flipped)

    def test_flip_sfen_with_promotions(self):
        sfen = "lnsgkgsn+B/9/p1pp1p1pp/6p2/9/2P6/P2P+RPPPP/2+r6/LN2KGSNL b SL3Pbgp 1"
        expected_flipped = "+Bnsgkgsnl/9/pp1p1pp1p/2p6/9/6P2/PPPP+RP2P/6+r2/LNSGK2NL b SL3Pbgp 1"
        self.assertEqual(flip_sfen(sfen), expected_flipped)


class TestFlipSfenSquare(unittest.TestCase):
    def test_flip_sfen_square(self):
        self.assertEqual(flip_sfen_square("7g"), "3c")
        self.assertEqual(flip_sfen_square("5e"), "5e")
        self.assertEqual(flip_sfen_square("9a"), "1i")
        self.assertEqual(flip_sfen_square("1i"), "9a")

    def test_flip_sfen_move_basic(self):
        self.assertEqual(flip_sfen_move("7g7f"), "3c3d")
        self.assertEqual(flip_sfen_move("2h3h"), "8b7b")
        self.assertEqual(flip_sfen_move("9a9b"), "1i1h")

    def test_flip_sfen_move_with_promotion(self):
        self.assertEqual(flip_sfen_move("7g7f+"), "3c3d+")
        self.assertEqual(flip_sfen_move("2h3h+"), "8b7b+")

    def test_flip_sfen_move_drop(self):
        self.assertEqual(flip_sfen_move("P*5e"), "P*5e")
        self.assertEqual(flip_sfen_move("R*2b"), "R*8h")

    def test_flip_sfen_square_edge_cases(self):
        # Test edge cases for the corners of the board
        self.assertEqual(flip_sfen_square("1a"), "9i")
        self.assertEqual(flip_sfen_square("9i"), "1a")
        self.assertEqual(flip_sfen_square("1i"), "9a")
        self.assertEqual(flip_sfen_square("9a"), "1i")

    def test_flip_sfen_square_invalid_input(self):
        # Test invalid inputs
        with self.assertRaises(ValueError):
            flip_sfen_square("0a")  # Invalid row
        with self.assertRaises(ValueError):
            flip_sfen_square("10a")  # Invalid row
        with self.assertRaises(ValueError):
            flip_sfen_square("1j")  # Invalid column
        with self.assertRaises(ValueError):
            flip_sfen_square("1")  # Missing column
        with self.assertRaises(ValueError):
            flip_sfen_square("a")  # Missing row
        with self.assertRaises(ValueError):
            flip_sfen_square("")  # Empty input


