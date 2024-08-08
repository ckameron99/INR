import utils

from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.distributions import kl_divergence, Normal
from torch.func import functional_call, stack_module_state
import torch.nn.functional as F
import torch
import numpy as np
import copy


class VariationalSirenLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        std_init,
        mu_magnitude,
        w0=30,
        c=6,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.mu = nn.Parameter(torch.FloatTensor(dim_in + 1, dim_out).uniform_(-mu_magnitude, mu_magnitude))
        self.log_std = nn.Parameter(torch.FloatTensor(dim_in + 1, dim_out).fill_(std_init))

        self.w0 = w0
        if activation is None:
            self.activation_function = lambda x: torch.sin(x * self.w0)
        else:
            self.activation_function = activation
        self.st = lambda x: F.softplus(x, beta=1, threshold=20)


    def forward(
        self,
        x,
    ):
        latent_sample = self.mu + self.st(self.log_std) * torch.randn_like(self.log_std)

        w, b = latent_sample[:-1], latent_sample[-1]

        return self.activation_function(x @ w + b)

    @property
    def std(self):
        return torch.exp(self.log_std)

    @std.setter
    def std(self, value):
        self.log_std.copy_(torch.log(value))


class ImageINR(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_fourier,
        dim_hidden,
        dim_out,
        num_layers,
        std_init,
        w0=30,
        c=6,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_fourier = dim_fourier

        layers = []
        for i in range(num_layers):
            w_std = (1 / dim_fourier) if i == 0 else (np.sqrt(c / dim_hidden) / w0)
            w_std *= 11 / 12
            layers.append(VariationalSirenLayer(
                dim_in = dim_fourier if i == 0 else dim_hidden,
                dim_out = dim_out if i == num_layers - 1 else dim_hidden,
                std_init = std_init,
                mu_magnitude = w_std,
                w0 = w0,
                c = c,
                activation= torch.nn.Identity() if i == num_layers - 1 else None
            ))

        self.layers = nn.Sequential(*layers)

    def forward(
        self,
        x,
    ):
        y = x
        for layer in self.layers:
            y = layer(y)
        return self.layers(x)


class Trainer(nn.Module):
    def __init__(
        self,
        size,
        dim_in,
        dim_fourier,
        dim_hidden,
        dim_out,
        num_layers,
        std_init,
        w0=30,
        c=6,
    ):
        super().__init__()
        representations = nn.ModuleList([
            ImageINR(
                dim_in=dim_in,
                dim_fourier=dim_fourier,
                dim_hidden=dim_hidden,
                dim_out=dim_out,
                num_layers=num_layers,
                std_init=std_init,
                w0=w0,
                c=c,
            ).to("cuda") for _ in range(size)  # complains if .to is removed :shrug:
        ])
        self.params, self.buffers = stack_module_state(representations)

        self.prior = ImageINR(
            dim_in=dim_in,
            dim_fourier=dim_fourier,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            num_layers=num_layers,
            std_init=std_init,
            w0=w0,
            c=c,
        )
        for i, layer in enumerate(self.prior.layers):
            w_std = (1 / dim_fourier) if i == 0 else (np.sqrt(c / dim_hidden) / w0)

            nn.init.constant_(layer.mu, 0)
            nn.init.constant_(layer.log_std, w_std * 0.5)  # init_std_scale param from COMBINER

        self.st = lambda x: F.softplus(x, beta=1, threshold=20)
        self.mse = torch.nn.MSELoss()
        self.base_model = [copy.deepcopy(self.prior).to("meta")]
        self.size = size


    def forward(
        self,
        X,
    ):
        X = torch.vmap(self.convert_posenc)(X)
        return torch.vmap(self.fmodel, randomness="different")(self.params, self.buffers, X)


    def convert_posenc(self, x):
        if x.dim() == 0:
            return torch.stack((x,)*32)
        assert self.prior.dim_fourier % (2 * self.prior.dim_in) == 0, "Embedding size must be the integer multiple of 2 * self.dim_in!"
        w = torch.exp(torch.linspace(0, np.log(1024), self.prior.dim_fourier // (2 * self.prior.dim_in), device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x

    def train(
        self,
        X,
        Y,
        epochs,
        lr,
        kl_beta,
    ):
        self.init_opt(lr, epochs=epochs)

        for epoch in range(epochs):
            self.opt.zero_grad()

            Y_hat = self(X)

            mse = self.mse(Y_hat, Y)

            kld = self.kld()

            loss = mse + kld * kl_beta

            loss.backward()
            self.opt.step()
            self.sched.step()

    def fmodel(self, params, buffers, x):
        return functional_call(self.base_model[0], (params, buffers), (x,))  # (x,)


    def init_opt(
        self,
        lr,
        epochs,
        train_prior=True,
    ):

        self.opt = Adam(self.params.values(), lr=lr)
        self.sched = lr_scheduler.MultiStepLR(self.opt, milestones=[int(epochs * 0.8)], gamma=0.5)


    def kld(self):
        prior_params = dict(self.prior.named_parameters())
        return sum([kl_divergence(
                Normal(self.params[f"layers.{i}.mu"], self.st(self.params[f"layers.{i}.log_std"])),
                Normal(prior_params[f"layers.{i}.mu"], prior_params[f"layers.{i}.log_std"])
            ).sum() for i in range(len(self.prior.layers))]) / self.size

    def update_prior(self):
        with torch.no_grad():
            prior_params = dict(self.prior.named_parameters())
            for i in range(len(self.prior.layers)):
                prior_params[f"layers.{i}.mu"].copy_(self.params[f"layers.{i}.mu"].clone().detach().mean(0))
                prior_params[f"layers.{i}.log_std"].copy_(torch.sqrt(
                    self.params[f"layers.{i}.mu"].clone().detach().var(0) + \
                    self.st(self.params[f"layers.{i}.log_std"].clone().detach()).pow(2).mean(0)
                ))

    def calculate_pnsr(self, X, Y):
        Y_hat = self(X)
        Y_hat = torch.clamp(Y_hat, 0., 1.)
        Y_hat = torch.round(Y_hat * 255) / 255
        return 20. * np.log10(1.) - 10. * (Y_hat - Y).detach().pow(2).mean().log10().cpu().item()

    def calculate_bpp(self, X, Y):
        return self.kld() / X.shape[1] / np.log(2)
