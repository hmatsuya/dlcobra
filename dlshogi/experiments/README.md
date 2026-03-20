# Experiments

Each experiment lives in its own directory: `expNNN_description/`.
Use underscores (not hyphens) so the folder is importable as a Python module.

## Structure

```
dlshogi/
├── config.yaml                # shared base config (all experiments inherit from this)
├── docs/                      # general documentation
│   ├── README.md              # documentation index
│   ├── optimization_ideas.md  # optimization strategies
│   └── PROFILING_GUIDE.md     # profiling guide
└── experiments/
    ├── __init__.py
    ├── log.md                 # experiment log (all experiments)
    ├── exp001_fewer_activations/
    │   ├── __init__.py
    │   ├── model.py           # experiment-specific network (optional)
    │   ├── config.yaml        # only overrides vs base config.yaml
    │   ├── run.sh             # launch script
    │   ├── resume.sh          # resume from last checkpoint
    │   ├── profile.py         # quick profiling (batch=128 default)
    │   ├── profile_detailed.py # detailed profiling (optional)
    │   ├── docs/              # experiment-specific documentation (optional)
    │   │   └── README.md      # profiling results, analysis
    │   └── fit.log            # training log (generated)
    └── exp002_xxx/
        ├── config.yaml
        └── run.sh
```

## Usage

Run an experiment:
```bash
bash dlshogi/experiments/exp001_fewer_activations/run.sh
```

Debug mode (2 epochs, no WandB):
```bash
bash dlshogi/experiments/exp001_fewer_activations/run.sh --debug
```

Extra CLI overrides:
```bash
bash dlshogi/experiments/exp001_fewer_activations/run.sh --trainer.max_epochs=10
```

Resume from last checkpoint (e.g. after early stopping):
```bash
bash dlshogi/experiments/exp001_fewer_activations/resume.sh
```

## Creating a new experiment

1. Copy an existing experiment directory (e.g. `exp001_fewer_activations/`)
2. Rename to `expNNN_short_description/`
3. Add `__init__.py` so the folder is importable as a Python package
4. Edit `config.yaml` with only the values that differ from the base `dlshogi/config.yaml`
5. If the experiment has a custom network, define it in `model.py` and set in `config.yaml`:
   ```yaml
   model:
     network: dlshogi.experiments.expNNN_short_description.model.PolicyValueNetwork
   ```
6. Run in debug mode to verify it starts correctly:
   ```bash
   bash dlshogi/experiments/expNNN_short_description/run.sh --debug
   ```
7. Run the profiler to check computation cost (uses batch=128 by default):
   ```bash
   bash dlshogi/experiments/expNNN_short_description/profile.sh
   ```
8. Add an entry to `log.md` (newest first) including the trainable parameter count:
   ```markdown
   ### expNNN: 説明
   **日付**: YYYY-MM-DD
   **ベース実験**: expMMM
   **パラメータ数**: X.XM
   **改善内容**:
   - ...
   ```
9. (Optional) For detailed profiling and analysis:
   ```bash
   python dlshogi/experiments/expNNN_short_description/profile_detailed.py
   ```
   Save results in `docs/` folder if experiment shows promise.

LightningCLI stacks configs: later `--config` files override earlier ones.
