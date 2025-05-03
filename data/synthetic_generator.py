import torch
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split

class SyntheticDatasetBuilder:
    """
    Generates toy datasets for binary classification:
    - blobs
    - circles
    - moons
    Returns train/test splits as torch.Tensors.
    """
    def __init__(self,
                 dataset_type: str = "blobs",
                 n_samples: int = 300,
                 noise: float = 0.1,
                 test_size: float = 0.2,
                 random_state: int = 42):
        assert dataset_type in ("blobs", "circles", "moons"), "dataset_type must be one of 'blobs','circles','moons'"
        self.dataset_type = dataset_type
        self.n_samples = n_samples
        self.noise = noise
        self.test_size = test_size
        self.random_state = random_state

    def generate(self):
        # produce numpy arrays
        if self.dataset_type == "blobs":
            X, y = make_blobs(n_samples=self.n_samples,
                              centers=2, random_state=self.random_state)
        elif self.dataset_type == "circles":
            X, y = make_circles(n_samples=self.n_samples,
                                noise=self.noise, random_state=self.random_state)
        else:  # moons
            X, y = make_moons(n_samples=self.n_samples,
                              noise=self.noise, random_state=self.random_state)

        # convert and split
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test
