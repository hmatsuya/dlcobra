# Project Structure

```
DeepLearningShogi/
├── dlshogi/                    # Python training code
│   ├── ptl.py                  # Main training entry (LightningCLI)
│   ├── config.yaml             # Base training configuration
│   ├── common.py               # Constants (features, move labels)
│   ├── data_loader.py          # HCPE/HDF5 data loading
│   ├── lr_scheduler.py         # Custom learning rate schedulers
│   ├── serializers.py          # Model save/load utilities
│   ├── network/                # Neural network architectures
│   │   ├── policy_value_network.py        # Network factory
│   │   ├── policy_value_network_resnet.py # ResNet implementation
│   │   ├── policy_value_network_senet.py  # SENet variant
│   │   └── ...
│   ├── experiments/            # Experiment directories
│   │   └── expNNN_description/ # Each experiment is a Python package
│   │       ├── __init__.py
│   │       ├── config.yaml     # Overrides base config
│   │       ├── model.py        # Custom network (optional)
│   │       └── run.sh          # Launch script
│   └── utils/                  # Data conversion & analysis tools
│       ├── csa_to_hcpe*.py     # CSA → HCPE converters
│       ├── hcpe_to_hdf5.py     # HCPE → HDF5 converter
│       └── stat_*.py           # Dataset statistics
│
├── cppshogi/                   # C++ Shogi library (Apery-based)
│   ├── position.cpp/hpp        # Board state management
│   ├── generateMoves.cpp/hpp   # Legal move generation
│   ├── bitboard.cpp/hpp        # Bitboard operations
│   ├── cppshogi.cpp/h          # Feature extraction for NN
│   └── python_module.cpp       # Python bindings
│
├── usi/                        # USI engine (game play)
├── selfplay/                   # MCTS self-play training
├── build/                      # C++ build output
└── Pipfile                     # Python dependencies
```

## Key Patterns

### Network Architecture
- All networks implement `PolicyValueNetwork` class
- Dual inputs: `x1` (board features), `x2` (hand pieces + flags)
- Dual outputs: policy logits (2187 moves), value logit (win prob)
- Factory function in `policy_value_network.py` handles instantiation

### Experiment Organization
- Directory naming: `expNNN_short_description/` (underscore, importable)
- Each experiment overrides only changed values from base config
- Custom networks referenced via fully-qualified class path
- See `dlshogi/experiments/README.md` for step-by-step creation guide

### Data Pipeline
- Raw: CSA/KIF game records
- Intermediate: HCPE (compressed positions + evaluations)
- Training: HDF5 (batched, memory-mapped)

### Feature Dimensions
- FEATURES1_NUM: 62 channels (piece positions, attacks)
- FEATURES2_NUM: 57 channels (hand pieces, check, nyugyoku)
- Board: 9×9 grid
- Move labels: 27 directions × 81 squares = 2187
