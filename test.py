import torch

a = torch.Tensor([1,2,3,4,5])
b = torch.Tensor([2,3,4,5,6])

mse = torch.nn.MSELoss(reduction='none')

print(mse(a, b))
