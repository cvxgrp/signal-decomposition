import numpy as np


def make_changepoint_basic():
    true = np.ones(1000) * 12
    true[400:] -= 1.5
    msk = np.logical_and(np.arange(1000) > 600, (np.arange(1000) // 100) % 2 == 0)
    true[msk] += 1
    msk = np.logical_and(np.arange(1000) > 750, (np.arange(1000) // 100) % 2 == 0)
    true[msk] += 1
    noise = np.random.randn(1000) * .15
    y = true + noise
    return y, np.c_[noise, true].T


def make_changepoint_random_drop():
    bcd_y, bcd_X_real = make_changepoint_basic()
    indices_full = list(range(len(bcd_X_real[1])))
    # select 20% randomly
    indices_20perc = np.random.choice(indices_full, int(.2 * len(indices_full)), replace=False)
    bcd_y_80perc = bcd_y.copy()
    bcd_y_80perc[indices_20perc] = np.nan
    return bcd_y_80perc, bcd_X_real


def make_changepoint_chunk_drop():
    bcd_y, bcd_X_real = make_changepoint_basic()
    indices_full = list(range(len(bcd_X_real[1])))
    # randomly select 5 chunks
    indices_chunks_start = np.random.choice(indices_full, 5, replace=False)
    bcd_y_chunks = bcd_y.copy()
    for e in indices_chunks_start:
        bcd_y_chunks[e:e + 25] = np.nan
    return bcd_y_chunks, bcd_X_real
