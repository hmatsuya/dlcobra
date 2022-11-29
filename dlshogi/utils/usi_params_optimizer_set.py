import optuna
from optuna import create_study, load_study
from optuna.pruners import MedianPruner
import argparse
import cshogi.cli
import sys
import logging

parser = argparse.ArgumentParser()
parser.add_argument('command1')
parser.add_argument('command2')
parser.add_argument('--options1', default='')
parser.add_argument('--options2', default='')
parser.add_argument('--study_name', default='mcts_params_optimizer')
parser.add_argument('--storage')
parser.add_argument('--trials', type=int, default=100)
parser.add_argument('--n_warmup_steps', type=int, default=30)
parser.add_argument('--games', type=int, default=100)
parser.add_argument('--byoyomi', type=int, default=1000)
parser.add_argument('--max_turn', type=int, default=320)
parser.add_argument('--opening')
parser.add_argument('--opening_moves', type=int, default=24)
parser.add_argument('--opening_seed', type=int, default=None)
parser.add_argument('--name')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

options_list = [{}, {}]
for i, kvs in enumerate([options.split(',') for options in (args.options1, args.options2)]):
    if len(kvs) == 1 and kvs[0] == '':
        continue
    for kv_str in kvs:
        kv = kv_str.split(':', 1)
        if len(kv) != 2:
            raise ValueError('options{} {}'.format(i + 1, kv_str))
        options_list[i][kv[0]] = kv[1]

def objective(trial):

    paramsetname = trial.suggest_categorical('paramsetname', ['high', 'low'])
    if paramsetname == 'high':
        paramset = {'C_init': 141, 'C_base': 29455, 'C_fpu_reduction': 32, 'C_init_root': 180, 'C_base_root': 34041, 'Softmax_Temperature': 120}
    else:
        paramset = {'C_init': 138, 'C_base': 30033, 'C_fpu_reduction': 32, 'C_init_root': 150, 'C_base_root': 26941, 'Softmax_Temperature': 137}

    params = {}
    suggested = {}
    for k, v in paramset.items():
        params[k] = v
        suggested[k] = v

    print('Trial {} start. paramsetname = {}, params = {}'.format(trial.number, paramsetname, str(suggested)))

    options1, options2 = options_list
    for k, v in suggested.items():
        options1[k] = v

    class Callback:
        def __init__(self):
            self.pruned = False

        def __call__(self, result):
            win_count = result['engine1_won']
            draw_count = result['draw']
            total_count = result['total']
            win_rate = (win_count + draw_count / 2) / total_count
            n = total_count - 1
            print('Trial {} game {} finished. win_count = {}, draw_count = {}, win_rate = {:.3f}'.format(trial.number, n, win_count, draw_count, win_rate))

            # 見込みのない最適化ステップを打ち切り
            trial.report(-win_rate, n)
            if trial.should_prune():
                self.pruned = True
                return False
            self.win_rate = win_rate
            return True

    # 先後入れ替えて対局
    callback = Callback()
    cshogi.cli.main(args.command1, args.command2, options1, options2, names=[args.name, None], games=args.games,
        mate_win=True, byoyomi=args.byoyomi, draw=args.max_turn,
        opening=args.opening, opening_moves=args.opening_moves, opening_seed=args.opening_seed, keep_process=True, is_display=False, debug=args.debug,
        callback=callback)

    if callback.pruned:
        raise optuna.TrialPruned()

    # 勝率を負の値で返す
    return -callback.win_rate

if args.storage:
    study = load_study(study_name=args.study_name, storage=args.storage, pruner=MedianPruner(n_warmup_steps=args.n_warmup_steps))
else:
    search_space = {'paramsetname': ['high', 'low']}
    study = create_study(sampler=optuna.samplers.GridSampler(search_space=search_space), pruner=MedianPruner(n_warmup_steps=args.n_warmup_steps))
study.optimize(objective, n_trials=args.trials)
