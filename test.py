from encoding import single_iREC, iREC
import torch
from matplotlib import pyplot as plt
import numpy as np
import random

class Args:
    def __init__(self):
        self.kl2_budget = 16
        self.seed_rec = 22

args = Args()

p_mu = torch.Tensor([3, 3, 3])
p_std = torch.Tensor([2, 2, 2])

q_mu = torch.Tensor([[3.5, 3.5, 3.5], [2.5, 2.5, 2.5]])
q_std = torch.Tensor([[1.8, 1.8, 1.8], [2.2, 2.2, 2.2]])

data = torch.Tensor([])

for _ in range(10000):
    samples, indexes = iREC(args, q_mu, q_std, p_mu, p_std)
    print(samples)
    exit()

    args.seed_rec = random.randint(1, 1e7)
    data = torch.cat((data, samples.T))

print(data)
print(data.mean(axis=0))
print(data.std(axis=0))


# Plotting a basic histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Basic Histogram')

# Display the plot
plt.show()
