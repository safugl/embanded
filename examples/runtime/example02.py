"""Example02."""
# pylint: skip-file
import time
import numpy as np
import compare_models
import pandas as pd


keys = ['EMB1', 'EMB2', 'EMB3', 'EMB4', 'RidgeCV1', 'RidgeCV2', 'RidgeCV3',
        'ARD', 'BRR', 'OLS', 'LassoCV']

sweep = [128, 1024, 16384, 65536]

df = []

for num_samples in sweep:

    np.random.seed(num_samples)

    num_dim = 128

    # Simulate two predictor groups
    F1 = np.random.randn(num_samples, num_dim//2)
    F2 = np.random.randn(num_samples, num_dim//2)

    F = [F1, F2]
    X = np.concatenate(F, axis=1)

    # Noise term
    N = np.random.randn(num_samples, 1)

    # Weights
    W1 = np.random.randn(num_dim//2, 1)/np.sqrt(num_dim//2)
    W2 = np.zeros((num_dim//2, 1))
    W = np.append(W1, W2, axis=0)

    y = (F1@W1 + F2@W2 + N).ravel()

    print('Number of samples: %i' % num_samples)

    for key in keys:
        start = time.time()
        if 'EMB' in key:
            W_i = compare_models.fit_model(key, F, y[:, None])
        else:
            W_i = compare_models.fit_model(key, X, y)
        time_elapsed_i = time.time() - start
        df.append(dict(num_samples=num_samples,
                  model=key,
                  time_elapsed=time_elapsed_i))

df = pd.DataFrame(df)


print('Model | ', end=' ')
for num_samples in sweep:
    print('%i samples | ' % num_samples, end=' ')
print('')
print(':-|:-|:-|:-|:-|:-|:-')
for key in keys:
    print('%s | ' % key, end=' ')
    for num_samples in sweep:
        time_elapsed, = df.loc[(df.model == key) & (
            df.num_samples == num_samples), 'time_elapsed'].to_numpy()
        print('%0.5f s | ' % time_elapsed, end=' ')
    print('')
