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


class Encoder(nn.Module):
    def __init__(self, trainer, kl_beta):
        super().__init__()
        self.trainer = trainer
        self.sample = self.gen_empty_sample(self.trainer.params)
        self.mask = copy.deepcopy(self.sample)
        self.groups = self.gen_groups(16, 25)  # TODO: make args parameters
        self.beta_list = [torch.ones_like(layer.mu) * kl_beta for layer in self.trainer.prior.layers]

    def train(
        self,
        X,
        Y,
        epochs,
        lr,
    ):
        for epoch in range(epochs):
            self.opt.zero_grad()

            Y_hat = self(X)

            mse = self.trainer.mse(Y_hat, Y)

            kld_beta = sum([(kl2 * beta).sum() for kl2, beta in zip(self.kld_list(), self.beta_list)])

            loss = mse + kld_beta

            loss.backward()
            self.opt.step()
            self.sched.step()

            if epoch > epochs // 10 and epoch % 100 == 0:  # TODO: make beta_adjust_interval a args
                self.adjust_beta_list()

    def adjust_beta_list(
        self,
    ):
        new_beta_list = [beta.clone() for beta in self.beta_list]
        kld_list = self.kld_list()
        for group in self.groups:
            group_kl_sum = sum([kld_list[layer_id].view(-1)[index].item() / np.log(2) for layer_id, index, _ in group])

            if group_kl_sum > (16 + 0.2):  # TODO: make parameters
                for layer_id, index, _ in group:
                    if self.mask[layer_id].view(-1)[index] == 0:
                       new_beta_list[layer_id].view(-1)[index] *= 1.1
            elif group_kl_sum < (16 - 0.2):  # TODO: make parameters
                for layer_id, index, _ in group:
                    if self.mask[layer_id].view(-1)[index] == 0:
                       new_beta_list[layer_id].view(-1)[index] /= 1.1
        self.beta_list = new_beta_list

    def merge_params(
        self,
        model_params,
        sample,
        mask,
    ):
        merged_params = dict()
        for k in model_params.keys():
            if "mu" in k:
                merged_params[k] = (1 - mask) * model_params[k] + mask * sample[k]
            elif "log_std" in k:
                merged_params[k] = (1 - mask) * model_params[k]
            else:
                raise ValueError(f"Unknown parameter key {k} encountered during merging")
        return merged_params

    def gen_empty_sample(
        self,
        model_params,
    ):
        sample = dict()
        for k in model_params.keys():
            if "mu" in k:
                sample[k] = torch.zeros_like(model_params[k])
            elif "log_std" in k:
                pass
            else:
                raise ValueError(f"Unknown parameter key {k} encountered during generation of empty sample")
        return sample

    def forward(
        self,
        X,
    ):
        X = torch.vmap(self.trainer.convert_posenc)(X)
        return torch.vmap(self.trainer.fmodel, randomness="different")(self.merge_params(self.trainer.params, self.sample, self.mask), {}, X)

    def init_opt(
        self,
        lr,
        epochs,
    ):
        self.opt = Adam(self.params.values(), lr=lr)
        self.sched = lr_scheduler.MultiStepLR(self.opt, milestones=[int(epochs * 0.8)], gamma=0.5)

    def gen_groups(
        self,
        kl2_budget,
        max_group_size,
    ):
        kld_list = self.kld_list()
        kl2_list = [kld / np.log(2) for kld in kld_list]
        kl2_cat = torch.cat(kl2_list)

        indices = torch.cat([
            torch.tensor([(layer_id, index) for index in range(len(tensor))]) for layer_id, tensor in enumerate(kl2_list)
        ]).to("cuda")  # TODO: args device

        indexed_kl2 = torch.vstack((indices.T, kl2_cat)).T

        shuffled_indexes = torch.randperm(len(indexed_kl2))
        shuffled_indexed_kl2 = indexed_kl2[shuffled_indexes]

        groups = current_group = []
        current_count = current_sum = 0

        for dim in shuffled_indexed_kl2:
            if current_sum + dim[2] > kl2_budget or current_count > max_group_size:
                groups.append(current_group)
                current_group = []
                current_sum = current_count = 0
            current_group.append(dim.tolist())
            current_sum += dim[2].item()
            current_count += 1
        if current_group:
            groups.append(current_group)

        return groups

    def kld_list(
        self,
    ):
        prior_params = dict(self.trainer.prior.named_parameters())
        kl2 = [kl_divergence(
                Normal(self.trainer.params[f"layers.{i}.mu"], self.trainer.st(self.trainer.params[f"layers.{i}.log_std"])),
                Normal(prior_params[f"layers.{i}.mu"], prior_params[f"layers.{i}.log_std"])
            ).mean(0).view(-1) for i in range(len(self.trainer.prior.layers))]
        return kl2

    def progressive_encode(
        self,
        X,
        Y,
        lr,
    ):
        for group_id, group in enumerate(self.groups):
            with torch.no_grad():
                sample_index = self.encode_group(group)
            num_tune = 100  # TODO: maybe call num_tune func?
            self.train(X, Y, num_tune, lr)
            if group_id % 10 == 0:
                self.adjust_beta_list()

    def encode_group(
        self,
        group,
    ):
        mu_q_rec = torch.zeros([len(group)])
        std_q_rec = torch.zeros([len(group)])
        mu_p_rec = torch.zeros([len(group)])
        std_p_rec = torch.zeros([len(group)])

        for i, (layer_id, index, _) in enumerate(group):
            row_size = self.trainer.prior.layers[layer_id].mu.size(1)
            row, col = index // row_size, index % row_size

            self.mask[layer_id][row, col] = 1
            mu_q_rec[i] = model.net[layer_id].mu.data[row, col]
            std_q_rec[i] = SmoothStd(model.net[layer_id].std.data)[row, col]
            mu_p_rec[i] = p_mu_list[layer_id][row, col]
            std_p_rec[i] = p_std_list[layer_id][row, col]
