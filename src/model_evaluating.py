from training_template.model_evaluator import ModelEvaluator


class IrisModelEvaluator(ModelEvaluator):
    def __init__(self):
        ...

    def evaluate_model(self) -> None:
        print("Evaluating model")
