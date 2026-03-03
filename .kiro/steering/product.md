# DeepLearningShogi (dlshogi)

A deep learning framework for Japanese chess (Shogi) based on AlphaGo/AlphaZero methodology.

## Purpose
- Train neural networks for Shogi move prediction and position evaluation
- Provide USI (Universal Shogi Interface) engine for competitive play
- Enable self-play training via Monte Carlo Tree Search (MCTS)

## Core Capabilities
- Policy network: predicts best moves (9×9×27 output labels)
- Value network: evaluates board positions (win probability)
- Dual-head architecture combining both networks
- Support for various ResNet-based architectures (10-30 blocks)

## Data Formats
- HCPE (Huffman Coded Position and Eval): compressed training data
- HCPE3: extended format with move visit counts for MCTS training
- HDF5: efficient storage for large datasets
- CSA/KIF: standard Shogi game record formats

## Target Users
- Shogi AI researchers and developers
- Competitive Shogi engine builders
- Deep learning practitioners interested in game AI
