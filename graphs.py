import numpy as np
import matplotlib.pyplot as plt


class Result:
    def __init__(self, data, label):
        self.data = np.array(data).T
        self.label = label

    def draw(self, plot):
        plot.plot(*self.data, label=self.label)
        plot.scatter(*self.data)

def draw(plot, save_file, *results):
    plot.xlabel("KL Divergence (bpp)")
    plot.ylabel("PSNR")
    for result in results:
        result.draw(plot)
    plot.legend()
    plot.grid(True)
    plot.savefig(save_file, dpi=300, bbox_inches="tight")
    plot.clf()

combiner = Result([
    [0.91, 25.525239676254035],
    [1.39, 28.190098156955706],
    [2.28, 31.12028104835409],
    [3.5, 33.600655034267305],
    [4.45, 34.71965262584059],

], label="COMBINER")

recombiner = Result([
    [ 0.297, 23.592],
    [ 0.719, 27.222],
    [ 0.938, 28.505],
    [ 1.531, 30.911],
    [ 1.922, 32.168],
    [ 3.344, 35.732],
    [ 4.391, 38.139],
], label="RECOMBINER")

coin = Result([
    [3.6, 25.8],
    [4.7, 27.5],
    [5.8, 29.3],
    [7.1, 30],
], label="COIN")

coin_pp = Result([
    [0.6, 24.1],
    [1.1, 26.8],
    [1.7, 28.8],
    [2.2, 30.5],
    [3.3, 32.3],
    [4.1, 33.8],
    [5.4, 35.6],

], label="COIN++")

own_combiner = Result([
    [1.81, 29.84],
    [2.70, 32],
    [2.44, 31.7]
], label="own_combiner")


def main():
    draw(plt, "figures/comb_comp.png", combiner, own_combiner)


if __name__ == "__main__":
    main()
