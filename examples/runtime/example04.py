"""Example04."""
# pylint: skip-file
import numpy as np
import torch
import compare_models
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dtype', help='dtype', default='float64')
args = parser.parse_args()


keys = ['EMB3 (PyTorch)', 'EMB4 (PyTorch)']


torch.jit.optimized_execution(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dtype == 'float64':
    dtype = torch.float64
elif args.dtype == 'float32':
    dtype = torch.float32
else:
    raise TypeError('Please check dtype')

print(dtype)

sweep = [128, 1024, 16384, 65536]

df = []
for P in sweep:

    num_dim = 512
    num_samples = 2048
    np.random.seed(num_samples)

    # Simulate two predictor groups
    F1 = np.random.randn(num_samples, num_dim//2)
    F2 = np.random.randn(num_samples, num_dim//2)

    F = [F1, F2]
    X = np.concatenate(F, axis=1)

    # Noise term
    N = np.random.randn(num_samples, P)

    # Weights
    W1 = np.random.randn(num_dim//2, P)/np.sqrt(num_dim//2)
    W2 = np.zeros((num_dim//2, P))
    W = np.append(W1, W2, axis=0)

    y = F1@W1 + F2@W2 + N

    y = torch.from_numpy(y).to(device=device, dtype=dtype)
    F = [torch.from_numpy(F[i]).to(device=device, dtype=dtype)
         for i in range(len(F))]

    del X

    if P == 128:
        __ = compare_models.fit_model(keys[0], F, y)

    for key in keys:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        W_i = compare_models.fit_model(key, F, y)

        end_event.record()
        torch.cuda.synchronize()
        time_elapsed_i = start_event.elapsed_time(end_event) / 1000

        df.append(dict(P=P,
                  model=key,
                  time_elapsed=time_elapsed_i))

        print('%s, %i, %0.3f' % (key, P, time_elapsed_i))

df = pd.DataFrame(df)


print('Model | ', end=' ')
for P in sweep:
    print('%i outcome varibles | ' % P, end=' ')
print('')
print(':-|:-|:-|:-|:-')
for key in keys:
    print('%s | ' % key, end=' ')
    for P in sweep:
        time_elapsed, = df.loc[(df.model == key) & (
            df.P == P), 'time_elapsed'].to_numpy()
        print('%0.5f s | ' % time_elapsed, end=' ')
    print('')
