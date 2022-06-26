from sklearn.model_selection import train_test_split
from training_template.data_preprocessor import DataPreprocessor


class IrisDataPreprocessor(DataPreprocessor):
    def __init__(self):
        ...

    def preprocess_data(self, data) -> None:
        X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"], test_size=0.33, random_state=42)
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
