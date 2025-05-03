import torch
import numpy as np
import matplotlib.pyplot as plt

class WeightNormMonitor:
    """
    Monitor the L2 norm of a model’s weights over training epochs,
    alongside the loss curve.
    """
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 num_epochs: int = 100):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.X_train = X_train
        self.y_train = y_train
        self.num_epochs = num_epochs
        self.weight_norms = []
        self.loss_history = []

    def track(self):
        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            logits = self.model(self.X_train)
            loss = self.loss_fn(logits, self.y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record total weight norm
            norms = [p.detach().norm().item() for p in self.model.parameters()]
            total_norm = np.linalg.norm(norms)
            self.weight_norms.append(total_norm)
            self.loss_history.append(loss.item())

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, WeightNorm={total_norm:.4f}")

        return self.loss_history, self.weight_norms

    def plot_weight_norm(self):
        plt.plot(self.weight_norms, marker='o', markersize=3)
        plt.title("L₂ Weight Norm over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Weight Norm")
        plt.grid(True)
        plt.show()

    def plot_loss(self):
        plt.plot(self.loss_history, marker='o', markersize=3)
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()
