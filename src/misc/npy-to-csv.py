import numpy as np
import csv

print('loading data...')
data = np.load('D:\\chess-data\\pgn\\scores(20million).npy', allow_pickle=True).item()
print('data loaded.')

log_interval = 10000

def normalize_cp(x: int):
    max_ = 5000
    min_ = -5000
    return max(min(2*(x - min_)/(max_ - min_) - 1, 1), -1)

with open('D:\\chess-data\\pgn\\master.csv', 'w', newline='') as out:
    outwriter = csv.writer(out)
    i = 0
    for FEN, score in data.items():
        i += 1
        if i % log_interval == 0:
            print("\rParsing Position [%d/%d]" % (i, len(data.items())), end='')

        score = normalize_cp(score) # normalize
        if FEN.split(' ')[1] == 'b': # flip for black
            score *= -1

        outwriter.writerow([FEN, score])
