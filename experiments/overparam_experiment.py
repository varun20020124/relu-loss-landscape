import pandas as pd
from experiments.shallow_trainer import ShallowReLUTrainer
from data.synthetic_generator import SyntheticDatasetBuilder

class OverparamExperiment:
    """
    Sweep over different hidden-layer widths and record final loss & test accuracy.
    """
    def __init__(self,
                 dataset_builder: SyntheticDatasetBuilder,
                 hidden_dims: list[int],
                 lr: float = 0.1,
                 num_epochs: int = 100):
        self.dataset_builder = dataset_builder
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.num_epochs = num_epochs
        self.results = pd.DataFrame(columns=["hidden_dim","final_loss","accuracy"])

    def run(self) -> pd.DataFrame:
        # Generate one dataset for all runs
        X_train, X_test, y_train, y_test = self.dataset_builder.generate()

        records = []
        for dim in self.hidden_dims:
            trainer = ShallowReLUTrainer(
                input_dim=X_train.shape[1],
                hidden_dim=dim,
                lr=self.lr,
                num_epochs=self.num_epochs
            )
            # override data
            trainer.X_train, trainer.X_test = X_train, X_test
            trainer.y_train, trainer.y_test = y_train, y_test

            trainer.train()
            final_loss = trainer.loss_history[-1]
            acc = trainer.evaluate()

            records.append({
                "hidden_dim": dim,
                "final_loss": final_loss,
                "accuracy": acc
            })

        self.results = pd.DataFrame.from_records(records)
        return self.results
