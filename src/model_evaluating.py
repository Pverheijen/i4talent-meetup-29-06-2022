from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from training_template.model_evaluator import ModelEvaluator


class IrisModelEvaluator(ModelEvaluator):
    def __init__(self):
        ...

    def evaluate_model(self, predictions, y_test) -> None:
        accuracy = accuracy_score(predictions, y_test)
        precision = precision_score(predictions, y_test, average="weighted")
        f1_score = f1_score(predictions, y_test, average="weighted")
        return {"accuracy": accuracy, "precision": precision, "f1_score": f1_score}
