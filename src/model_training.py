from sklearn.linear_model import LogisticRegression
from training_template.model_trainer import ModelTrainer


class IrisModelTrainer(ModelTrainer):
    def __init__(self):
        ...

    def train_model(self, data) -> None:
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(data["X_train"], data["y_train"])
        return logreg
