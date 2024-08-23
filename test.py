import torch
import numpy as np
import operator

"""
(tqdm(range(iters)) if verbose else range(iters))
def update_best(self, tune_beta, loss, X, Y):
    if tune_beta:
        metric = self.calculate_pnsr(X, Y)
        op = operator.le
        m = torch.max
    else:
        metric = loss
        op = operator.ge
        m = torch.min
    with torch.no_grad():
        for k in self.params.keys():
            self.best_params[k][1 ^ op(metric, self.best_metric)] = self.params[k][1 ^ op(metric, self.best_metric)].detach()
        self.best_metric = m(self.best_metric, metric.detach())"""

def update_best(maximise, a, best, best_params):
    if maximise:
        metric = a
        op = operator.le
        m = torch.fmax
    else:
        metric = a
        op = operator.ge
        m = torch.fmin
    with torch.no_grad():
        best_params[1^op(metric, best)] = a[1^op(metric, best)]
        best = m(best, metric)
    return best, best_params


best, best_params = torch.empty(5).fill_(np.nan), torch.empty(5).fill_(np.nan)
for _ in range(15):
    best, best_params = update_best(maximise=True, a=torch.rand(5), best=best, best_params=best_params)

print(best, best_params)
