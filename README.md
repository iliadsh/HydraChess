![Hydra](https://i.ibb.co/ChY2NnB/hydra-chess.png)

--------------------------------------------------------------------------------

# About
UCI compatible deep neural network chess engine.

Hydra contains two main components:
- Parallel Monte-Carlo tree search.
- NN position evaluator.

The project is built on:
- The [libchess](https://github.com/Mk-Chan/libchess) library for board representation and move generation.
- Facebook's [PyTorch](https://github.com/pytorch/pytorch) library for model backend.
- The AlphaZero paper: [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)

The weights found in the repository were trained with the Kaggle [Finding Elo](https://www.kaggle.com/c/finding-elo/data) dataset.
# Building
First install:
- CUDA v10.2
- cuDNN v8.0.2
- libtorch v1.6.0 

Then run:
```
git clone https://github.com/ABigPickle/HydraChess.git
cd HydraChess
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build .
```
# Supervised Learning
Just run with:
```
./HydraChess -train </path/to/dataset.csv>
```
Which will train the network with the parameters defined the `config.hpp`. The dataset should be formatted as a CSV (comma seperated values) file like so:
```
fen,evaluation
fen,evaluation
...
fen,evaluation
```
Where each line represents a training example. The `fen` string should be a game state in [FEN format](hhttps://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) and the `evaluation` should be a single decimal value from `[-1, 1]` where `1` is winning for white and vice versa.
# Reinforcement Learning
Reinforcement learning is currently not implemented. If you want to take a crack at it, please submit a pull request!