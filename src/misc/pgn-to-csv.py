#!/usr/bin/env python3

import sys
import chess.pgn, chess.engine
import csv
import os

class PgnToCsv:
    def __init__(self,
                 engine_path,
                 pgn_path):
        self.engine_path = engine_path
        self.pgn_path = pgn_path
    
    def normalize_cp(self, x: int):
        max_ = 5000
        min_ = -5000
        return max(min(2*(x - min_)/(max_ - min_) - 1, 1), -1)

    def run(self):
        log_interval = 1
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

        with open(self.pgn_path) as pgn, open(os.path.splitext(self.pgn_path)[0]+ '-dataset.csv', 'w', newline='') as out:
            outwriter = csv.writer(out)

            gn = 0
            while game := chess.pgn.read_game(pgn):
                gn += 1

                if gn % log_interval == 0:
                    print('\rParsing Game #' + str(gn), end='')

                board = chess.Board()

                for move in game.mainline_moves():
                    board.push(move)
                    fen = board.fen()
                    info = engine.analyse(board, chess.engine.Limit(depth = 8))
                    value = info['score'].white().score(mate_score=10000)
                    if board.turn == chess.BLACK:
                        value *= -1
                    value = self.normalize_cp(value)
                    outwriter.writerow([fen, value])

        engine.quit()

        print('\nDone.')

if __name__ == "__main__":
    runner = PgnToCsv('D:\\stockfish\\Windows\\stockfish_64.exe',
                      'D:\\chess-data\\pgn\\kaggle.pgn')
    runner.run()
