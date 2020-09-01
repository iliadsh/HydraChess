![Hydra](https://i.ibb.co/ChY2NnB/hydra-chess.png)

--------------------------------------------------------------------------------

# About
UCI compatible deep neural network chess engine.

Hydra contains two main components:
- Parallel UCT search.
- NN position evaluator.

The project is built on:
- The [libchess](https://github.com/Mk-Chan/libchess) library for board representation and move generation.
- Facebook's [PyTorch](https://github.com/pytorch/pytorch) library for model backend.
- The AlphaZero paper: [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- Arman Maesumi's paper on [Playing Chess with Limited Look Ahead](https://arxiv.org/abs/2007.02130)
- And various other great internet resources.

The weights found in the repository were trained with 21 million depth 12 Stockfish evaluated positions provided by Mr Maesumi.
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
Where each line represents a training example. The `fen` string should be a game state in [FEN format](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) and the `evaluation` should be a single decimal value from `[-1, 1]` where `1` is winning for the **current side to move** and vice versa.
# Reinforcement Learning
Reinforcement learning is currently not implemented. If you want to take a crack at it, please submit a pull request!
# Results
After 25 epochs of training the model on the Stockfish evaluation dataset, the estimated play strength is about 1600 elo.

(Intel i7-7700k, Nvidia GTX 1060 6gb, 2 threads, 16000 iterations per-thread)

Engine (black) against Stockfish level 4 (1600-1700 elo):

![img](https://i.imgur.com/N5LSxoF.gif)

Engine (black) against Stockfish level 3 (1300-1400 elo):

![img](https://i.imgur.com/lzT203y.gif)