import unittest
from dlshogi.utils.plaintext_to_hcpe import flip_sfen

class TestFlipSfen(unittest.TestCase):
    def test_flip_sfen_basic(self):
        sfen = "lnsgk2nl/1r4gs1/p1pppp1pp/1p4p2/7P1/2P6/PP1PPPP1P/1SG4R1/LN2KGSNL b Bb 1"
        expected_flipped = "ln2kgsnl/1sg4r1/pp1pppp1p/2p4p1/1P7/6P2/P1PPPP1PP/1R4GS1/LNSGK2NL b Bb 1"
        self.assertEqual(flip_sfen(sfen), expected_flipped)

    def test_flip_sfen_with_promotions(self):
        sfen = "lnsgkgsn+B/9/p1pp1p1pp/6p2/9/2P6/P2P+RPPPP/2+r6/LN2KGSNL b SL3Pbgp 1"
        expected_flipped = "+Bnsgkgsnl/9/pp1p1pp1p/2p6/9/6P2/PPPP+RP2P/6+r2/LNSGK2NL b SL3Pbgp 1"
        self.assertEqual(flip_sfen(sfen), expected_flipped)