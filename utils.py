import atexit
import hashlib
import time
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import torch


def checksum(t):
    torch.manual_seed(69)
    t = torch.flatten(t)
    tb = t.cpu().detach().numpy().tobytes()
    sha256=hashlib.sha256()
    sha256.update(tb)
    return sha256.hexdigest()

def cum_dist(model):
    n = len(model.prior.layers)
    fig, axes = plt.subplots(n, 1, figsize=(5, 5*n))

    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        data = model.params[f"layers.{i}.mu"].clone().detach().cpu().flatten()
        count, bins_count = np.histogram(data, bins=100)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        ax.plot(bins_count[1:], cdf)
        ax.set_title(f"pdf for layer {i}")

    plt.tight_layout()
    plt.show()

class MethodTimeTracker:
    def __init__(self):
        self.cumulative_time = 0

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.cumulative_time += elapsed_time
            return result

        return wrapper

    def report(self, method_name):
        print(f"Total time spent in {method_name}: {self.cumulative_time:.4f} seconds")

def time_tracking_decorator(method):
    tracker = MethodTimeTracker()
    atexit.register(tracker.report, method.__name__)
    return tracker(method)
