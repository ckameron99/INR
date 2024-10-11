import graphs

import numpy as np
import matplotlib.pyplot as plt

with open("results5.txt", "r") as f:
    data = f.read()

data = data.split("\n")
bpps = [float(num) for num in data[2::4]]
psnrs = [float(num[7:13]) for num in data[1::4]]
data = list(zip(bpps, psnrs))
data.sort(key=lambda x: x[0])
print(data)
combiner = graphs.Result(
    data,
    label="own_combiner"
)

graphs.draw(plt, "figures/comb_comp5.png", graphs.combiner, combiner)
