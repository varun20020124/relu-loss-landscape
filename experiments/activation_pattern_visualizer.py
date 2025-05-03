import torch
import numpy as np
import matplotlib.pyplot as plt

class ActivationPatternVisualizer:
    """
    Visualize how a shallow ReLU network partitions 2D input space
    by neuron activation patterns.
    """
    def __init__(self, model, xlim=(-3,3), ylim=(-3,3), resolution=200):
        self.model = model
        self.xlim = xlim
        self.ylim = ylim
        self.resolution = resolution
        self.patterns = None
        self.grid = None

    def compute_patterns(self):
        # build grid
        xs = np.linspace(self.xlim[0], self.xlim[1], self.resolution)
        ys = np.linspace(self.ylim[0], self.ylim[1], self.resolution)
        Xg, Yg = np.meshgrid(xs, ys)
        pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
        
        # forward pass of hidden activations
        with torch.no_grad():
            inp = torch.tensor(pts, dtype=torch.float32)
            hidden = self.model.hidden_layer(inp)  # (N, hidden_dim)
            activations = (hidden > 0).cpu().numpy().astype(int)  # binary mask

        # encode each pattern as integer for coloring
        codes = np.packbits(activations, axis=1)  # pack bits along each row
        codes = codes.flatten()  # one code per point

        self.patterns = codes.reshape(self.resolution, self.resolution)
        self.grid = (Xg, Yg)

    def plot_partition(self, cmap='tab20'):
        if self.patterns is None:
            raise RuntimeError("Call compute_patterns() first.")
        Xg, Yg = self.grid
        plt.figure(figsize=(6,6))
        plt.pcolormesh(Xg, Yg, self.patterns, cmap=cmap, shading='auto')
        plt.title("Activation‐Pattern Partitioning")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.show()
