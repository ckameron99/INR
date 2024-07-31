import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
torch.manual_seed(0)

# Here's a simple MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
    
device = 'cuda'
num_models = 10

data = torch.randn(100, 64, 1, 28, 28, device=device)
targets = torch.randint(10, (6400,), device=device)

models = [SimpleMLP().to(device) for _ in range(num_models)]

minibatches = data[:num_models]
predictions_diff_minibatch_loop = [model(minibatch) for model, minibatch in zip(models, minibatches)]

minibatch = data[0]
predictions2 = [model(minibatch) for model in models]

from torch.func import stack_module_state

params, buffers = stack_module_state(models)

from torch.func import functional_call
import copy

# Construct a "stateless" version of one of the models. It is "stateless" in
# the sense that the parameters are meta Tensors and do not have storage.
base_model = copy.deepcopy(models[0])
base_model = base_model.to('meta')

def fmodel(params, buffers, x):
    return functional_call(base_model, (params, buffers), (x,))

#print([p.size(0) for p in params.values()]) # show the leading 'num_models' dimension

assert minibatches.shape == (num_models, 64, 1, 28, 28) # verify minibatch has leading dimension of size 'num_models'

from torch import vmap

predictions1_vmap = vmap(fmodel)(params, buffers, minibatches)

# verify the ``vmap`` predictions match the
assert torch.allclose(predictions1_vmap, torch.stack(predictions_diff_minibatch_loop), atol=1e-3, rtol=1e-5)

predictions2_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, minibatch)

assert torch.allclose(predictions2_vmap, torch.stack(predictions2), atol=1e-3, rtol=1e-5)

optim = Adam(params.values(), 2e-4)