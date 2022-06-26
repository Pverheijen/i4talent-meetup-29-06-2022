from sklearn.datasets import load_iris
from training_template.data_loader import DataLoader


class IrisDataLoader(DataLoader):
    def __init__(self):
        ...

    def load_data(self):
        iris_dataset = load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        return {"X": X, "y": y}
