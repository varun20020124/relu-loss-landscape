import torch
import numpy as np
import matplotlib.pyplot as plt

class GradientTrajectoryAnalyzer:
    """
    Train a model while recording the gradient norm at each epoch.
    """
    def __init__(self, model, loss_fn, optimizer, X_train: torch.Tensor, y_train: torch.Tensor,
                 num_epochs: int = 100):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.X_train = X_train
        self.y_train = y_train
        self.num_epochs = num_epochs
        self.grad_norms = []
        self.loss_history = []

    def track(self):
        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            logits = self.model(self.X_train)
            loss = self.loss_fn(logits, self.y_train)
            self.optimizer.zero_grad()
            loss.backward()

            # record gradient norm
            grads = []
            for p in self.model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().view(-1))
            grad_vec = torch.cat(grads)
            grad_norm = grad_vec.norm().item()
            self.grad_norms.append(grad_norm)
            self.loss_history.append(loss.item())

            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, ‖∇L‖={grad_norm:.4f}")

        return self.loss_history, self.grad_norms

    def plot_grad_norm(self):
        plt.plot(self.grad_norms, marker='o', markersize=3)
        plt.title("Gradient Norm over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("‖Gradient‖")
        plt.grid(True)
        plt.show()

    def plot_loss(self):
        plt.plot(self.loss_history, marker='o', markersize=3)
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()
