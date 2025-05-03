import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from models.shallow_relu import ShallowReLU

class ShallowReLUTrainer:
    """
    Trainer for a 1-hidden-layer ReLU model on a binary blob dataset.
    """
    def __init__(self, input_dim=2, hidden_dim=2, lr=0.1, num_epochs=100):
        self.model = ShallowReLU(input_dim, hidden_dim)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.loss_history = []

    def load_data(self, n_samples=300, test_size=0.2, random_state=42):
        X, y = make_blobs(n_samples=n_samples, centers=2,
                          n_features=self.model.hidden_layer.in_features,
                          random_state=random_state)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(
             X, y, test_size=test_size, random_state=random_state
        )

    def train(self):
        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            logits = self.model(self.X_train)
            loss = self.criterion(logits, self.y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

    def plot_loss_curve(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_history, marker='o', markersize=3)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_test)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            acc = (preds == self.y_test).float().mean().item()
        print(f"Test Accuracy: {acc*100:.2f}%")
        return acc
