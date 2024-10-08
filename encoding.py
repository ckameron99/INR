import torch
from scipy.stats import norm
from torch.distributions import Exponential, Normal
from torch.quasirandom import SobolEngine


def prior_samples(n_samples, n_variable, seed):
    sobol = SobolEngine(n_variable, scramble=True, seed=seed)
    samples_sobol = sobol.draw(n_samples)
    samples_i = torch.from_numpy(norm.ppf(samples_sobol))
    samples_i = torch.clamp(samples_i, -10, 10)
    return samples_i

def iREC(args, mu_q, std_q, mu_p, std_p):
    res = [single_iREC(args, single_mu_q, single_std_q, mu_p, std_p) for (single_mu_q, single_std_q) in zip(mu_q, std_q)]

    sample = torch.stack([r[0] for r in res])
    index = torch.Tensor([r[1] for r in res])

    return sample, index

def single_iREC(args, mu_q, std_q, mu_p, std_p):
    # fixed bit budget
    n_samples = 2 ** args.kl2_budget
    n_variable = mu_p.shape[0]
    N = 1
    exp_dist = Exponential(torch.tensor(1.0))
    deltas = exp_dist.sample((N, n_samples))
    ts = torch.cumsum(deltas, dim=1)
    gumbels = - torch.log(ts)

    normal_dist = Normal(mu_p * torch.ones(N, n_samples, n_variable),
                        std_p * torch.ones(N, n_samples, n_variable))

    xs = prior_samples(n_samples, n_variable, args.seed) * std_p + mu_p
    xs = xs.unsqueeze(0)

    q_normal_dist = Normal(mu_q, std_q)
    log_ratios = q_normal_dist.log_prob(xs) - normal_dist.log_prob(xs)
    log_ratios = log_ratios.sum(dim=-1)

    # importance weight perturbed with ordered Gumbel noise
    perturbed = log_ratios + gumbels

    argmax_indices = torch.argmax(perturbed, dim=1)
    approx_samp_mask = torch.zeros(N, n_samples, dtype=torch.bool, device=argmax_indices.device)
    approx_samp_mask.scatter_(1, argmax_indices.unsqueeze(1), True)

    approx_samps = xs[approx_samp_mask]
    received_samples = approx_samps
    return received_samples[0], argmax_indices.item()
