import torch
import numpy as np
import matplotlib.pyplot as plt

class GradientAnalyzer:
    """
    Compute and visualize gradients and Hessians of a model's loss w.r.t. its parameters.
    """
    def __init__(self, model, loss_fn, X: torch.Tensor, y: torch.Tensor):
        self.model = model
        self.loss_fn = loss_fn
        self.X = X
        self.y = y
    
    def compute_condition_number(self, H: np.ndarray):
        eigs = np.linalg.eigvalsh(H)
        min_eig = np.min(np.abs(eigs))  # Avoid division by zero
        if min_eig < 1e-8:
            return np.inf  # Ill-conditioned matrix
        return np.max(np.abs(eigs)) / min_eig

    def compute_gradient(self):
        """
        Returns a flat numpy array of the gradient of loss on (X, y) w.r.t. all model parameters.
        """
        self.model.zero_grad()
        logits = self.model(self.X)
        loss = self.loss_fn(logits, self.y)
        loss.backward()
        grads = []
        for p in self.model.parameters():
            grads.append(p.grad.detach().cpu().view(-1))
        grad_vector = torch.cat(grads).numpy()
        return grad_vector

    def compute_hessian(self):
        """
        Returns (approximate) Hessian matrix using double‑loop finite differences.
        For small models only!
        """
        eps = 1e-3
        params = [p for p in self.model.parameters()]
        # flatten and record original vector
        orig = torch.cat([p.detach().view(-1) for p in params])
        n = orig.numel()
        H = np.zeros((n, n), dtype=float)

        for i in range(n):
            # +epsilon
            orig_i = orig[i].item()
            orig[i] = orig_i + eps
            self._load_from_flat(orig)
            g_plus = self.compute_gradient()

            # -epsilon
            orig[i] = orig_i - eps
            self._load_from_flat(orig)
            g_minus = self.compute_gradient()

            # restore
            orig[i] = orig_i
            self._load_from_flat(orig)

            H[:, i] = (g_plus - g_minus) / (2 * eps)

        return H

    def _load_from_flat(self, flat_vec: torch.Tensor):
        """
        Utility to write a flat parameter vector back into model.parameters().
        """
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(flat_vec[offset:offset+numel].view_as(p))
            offset += numel

    def plot_gradient_norm(self):
        grad = self.compute_gradient()
        plt.bar(range(len(grad)), np.abs(grad))
        plt.title("Gradient Magnitudes")
        plt.xlabel("Parameter Index")
        plt.ylabel("|Gradient|")
        plt.show()

    # def plot_hessian_spectrum(self, H: np.ndarray):
    #     eigs = np.linalg.eigvalsh(H)
    #     plt.plot(np.sort(eigs)[::-1], marker='o')
    #     plt.title("Hessian Eigenvalues")
    #     plt.xlabel("Index")
    #     plt.ylabel("Eigenvalue")
    #     plt.grid(True)
    #     plt.show()
    def plot_hessian_spectrum(self, H: np.ndarray):
        eigs = np.linalg.eigvalsh(H)
        
        # Filter out near-zero eigenvalues to avoid numerical instability
        eigs_positive = eigs[eigs > 1e-6]
        
        if len(eigs_positive) == 0:
            kappa = float('inf')
        else:
            kappa = eigs_positive.max() / eigs_positive.min()
        
        # Plot eigenvalues
        plt.plot(np.sort(eigs)[::-1], marker='o')
        plt.title(f"Hessian Eigenvalues (κ ≈ {kappa:.2f})")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.grid(True)
        plt.show()
        
        print(f"Condition number (κ): {kappa:.2f}")

