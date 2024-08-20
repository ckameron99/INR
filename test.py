import torch
import numpy as np

a = torch.rand(5,5,5)
b = torch.rand(5)
print(20. * np.log10(1.) - 10. * a.detach().pow(2).mean((1,2)).log10())
