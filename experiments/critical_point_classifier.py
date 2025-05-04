import torch
import numpy as np
from analysis.gradient_analysis import GradientAnalyzer

class CriticalPointClassifier:
    """
    Classify a model’s current parameter point as
    local minimum / maximum / saddle (or not critical).
    """
    def __init__(self, model, loss_fn, X: torch.Tensor, y: torch.Tensor):
        self.model = model
        self.loss_fn = loss_fn
        self.X = X
        self.y = y
        self.analyzer = GradientAnalyzer(model, loss_fn, X, y)

    def classify_current(self, grad_tol: float = 1e-3, eig_tol: float = 1e-3):
        # 1) gradient norm
        grad = self.analyzer.compute_gradient()
        grad_norm = np.linalg.norm(grad)

        # 2) Hessian eigenvalues
        H = self.analyzer.compute_hessian()
        eigs = np.linalg.eigvalsh(H)

        # 3) classification
        if grad_norm > grad_tol:
            label = "Not critical (‖∇L‖ > tol)"
        else:
            num_pos = np.sum(eigs > eig_tol)
            num_neg = np.sum(eigs < -eig_tol)
            if num_neg == 0 and num_pos > 0:
                label = "Local minimum"
            elif num_pos == 0 and num_neg > 0:
                label = "Local maximum"
            else:
                label = "Saddle point"

        return label, eigs
