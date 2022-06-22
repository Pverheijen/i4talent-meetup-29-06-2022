from abc import ABC, abstractmethod


class ModelEvaluator(ABC):
    @abstractmethod
    def evaluate_model(self) -> None:
        """Method -to be implemented- evaluates the model."""
