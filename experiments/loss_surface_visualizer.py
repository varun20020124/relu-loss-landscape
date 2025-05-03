import torch
import numpy as np
import matplotlib.pyplot as plt

class LossSurfaceVisualizer:
    """
    Sweeps two model parameters over a grid and records the loss, then plots a 2D contour.
    """
    def __init__(self, model, loss_fn, X: torch.Tensor, y: torch.Tensor):
        self.model = model
        self.loss_fn = loss_fn
        self.X = X
        self.y = y
        self.loss_grid = None
        self.grid = None
        self.param1_name = ""
        self.param2_name = ""

    def _get_param(self, full_name: str):
        # Navigate attributes (e.g. "hidden_layer.weight")
        obj = self.model
        for attr in full_name.split('.'):
            obj = getattr(obj, attr)
        return obj

    def sweep_2d(self,
                 param1: tuple,
                 param2: tuple,
                 resolution: int = 50,
                 span: float = 1.0):
        """
        param1, param2: (param_name, index_tuple)
        resolution: grid size per axis
        span: how far around the original value to sweep (±span)
        """
        name1, idx1 = param1
        name2, idx2 = param2
        p1 = self._get_param(name1)
        p2 = self._get_param(name2)

        # record names for plotting
        self.param1_name = f"{name1}{idx1}"
        self.param2_name = f"{name2}{idx2}"

        # get original values
        orig1 = p1.data[idx1].item()
        orig2 = p2.data[idx2].item()

        # build grid
        g1 = np.linspace(orig1 - span, orig1 + span, resolution)
        g2 = np.linspace(orig2 - span, orig2 + span, resolution)
        losses = np.zeros((resolution, resolution), dtype=float)

        # sweep
        for i, v1 in enumerate(g1):
            for j, v2 in enumerate(g2):
                with torch.no_grad():
                    p1.data[idx1] = v1
                    p2.data[idx2] = v2
                    logits = self.model(self.X)
                    losses[j, i] = self.loss_fn(logits, self.y).item()

        # restore originals
        with torch.no_grad():
            p1.data[idx1] = orig1
            p2.data[idx2] = orig2

        self.loss_grid = losses
        self.grid = (g1, g2)
        return losses

    def plot_surface(self):
        """
        Requires that sweep_2d has been called first.
        """
        if self.loss_grid is None:
            raise RuntimeError("Call sweep_2d before plot_surface!")

        g1, g2 = self.grid
        Xg, Yg = np.meshgrid(g1, g2)
        plt.contourf(Xg, Yg, self.loss_grid, levels=50)
        plt.colorbar(label="Loss")
        plt.xlabel(self.param1_name)
        plt.ylabel(self.param2_name)
        plt.title("Loss Surface Contour")
        plt.show()
