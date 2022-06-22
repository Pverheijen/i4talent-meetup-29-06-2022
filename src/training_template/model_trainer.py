from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    @abstractmethod
    def train_model(self) -> None:
        """Method -to be implemented- that trains the model."""
