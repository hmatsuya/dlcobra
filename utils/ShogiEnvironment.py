from shogi import Ayane
import cshogi
import statistics

class ShogiEnvironment:
    def __init__(self, args={}):
        self.winner = None
        self.engines = []
        self.actions = []
    
        for path in ['../x64/Release/dlshogi_tensorrt.exe', '../x64/Release/dlshogi_tensorrt.exe']:
            e = Ayane.UsiEngine()
            e.connect(path)
            engines.append(e)

        self.board = cshogi.Board()


    def __str__(self):
        return ''

    #
    # Should be defined in all games
    #
    def reset(self, args={}):
        # raise NotImplementedError()
        self.board.set_sfen(cshogi.STARTING_SFEN)
        self.winner = None
        self.actions = []

    #
    # Should be defined in all games except you implement original step() function
    #
    def play(self, action, player):
        # raise NotImplementedError()
        assert(not board.is_game_over())
        self.board.push_usi(self.actions[action])
        if board.is_game_over():
            self.winner = player
        else:
            for e in self.engines:
                e.usi_position(self.board.sfen)
                e.usi_go_and_wait_bestmove()

    #
    # Should be defined in games which has simultaneous trainsition
    #
    def step(self, actions):
        for p, action in actions.items():
            if action is not None:
                self.play(action, p)

    #
    # Should be defined if you use multiplayer sequential action game
    #
    def turn(self):
        return self.board.turn()

    #
    # Should be defined if you use multiplayer simultaneous action game
    #
    def turns(self):
        return [self.turn()]

    #
    # Should be defined in all games
    #
    def terminal(self):
        # raise NotImplementedError()
        return self.board.is_game_over()

    #
    # Should be defined if you use immediate reward
    #
    def reward(self):
        return {}

    #
    # Should be defined in all games
    #
    def outcome(self):
        # raise NotImplementedError()
        assert(self.board.is_game_over())
        return self.winner

    #
    # Should be defined in all games
    #
    def legal_actions(self, player):
        raise NotImplementedError()

    #
    # Should be defined in all games
    #
    def action_length(self):
        raise NotImplementedError()

    #
    # Should be defined if you use multiplayer game or add name to each player
    #
    def players(self):
        return [0, 1]

    #
    # Should be defined in all games
    #
    def observation(self, player=None):
        raise NotImplementedError()

    #
    # Should be defined if you encode action as special string
    #
    def action2str(self, a, player=None):
        return str(a)

    #
    # Should be defined if you encode action as special string
    #
    def str2action(self, s, player=None):
        return int(s)

    #
    # Should be defined if you use network battle mode
    #
    def diff_info(self, player=None):
        return ''

    #
    # Should be defined if you use network battle mode
    #
    def update(self, info, reset):
        raise NotImplementedError()
