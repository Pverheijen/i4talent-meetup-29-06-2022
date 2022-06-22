from training_template.data_loader import DataLoader
from sklearn.datasets import load_iris

class IrisDataLoader(DataLoader):
    def __init__(self):
        ...

    def load_data(self):
        print("Loading data")
        iris_dataset = load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        return {"X": X, "y": y}
        
