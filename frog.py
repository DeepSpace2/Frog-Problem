import random
import numpy as np
import pandas as pd

from collections import defaultdict
from matplotlib import pyplot as plt
from multiprocessing.pool import Pool
from scipy.interpolate import make_interp_spline


TRIALS = 500_000


def jump(leaves):
    pos = leaves[0]
    leaps = 0
    while pos < leaves[-1]:
        pos += random.randint(1, leaves[-1] - pos)
        leaps += 1
    return leaps


def plot(df, num_of_leaves, trials):
    fig, ax = plt.subplots()
    jumps_smooth = np.linspace(df.jumps.min(), df.jumps.max())
    spl = make_interp_spline(df.jumps.tolist(), df.occurrences.tolist())
    occurrences_smooth = spl(jumps_smooth)
    ax.plot(jumps_smooth, occurrences_smooth)
    ax.set(xlabel='Jumps', ylabel='Occurrences', title=f'Num of leaves = {num_of_leaves}\nTrials = {trials}')
    plt.savefig(f'graphs/{num_of_leaves}_{trials}')
    plt.close()


def start(num_of_leaves):
    print(f'Calculating for {num_of_leaves} leaves')
    leaves = range(1, num_of_leaves + 1)
    data = defaultdict(int)

    for trial in range(TRIALS):
        data[jump(leaves)] += 1

    df = pd.DataFrame(data=list(data.items()), columns=['jumps', 'occurrences'])
    df.sort_values('jumps', inplace=True)
    plot(df, num_of_leaves, TRIALS)


if __name__ == '__main__':
    Pool(4).map(start, range(5, 1001))
