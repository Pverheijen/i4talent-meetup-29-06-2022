from abc import ABC, abstractmethod


class DataPreprocessor(ABC):
    @abstractmethod
    def preprocess_data(self) -> None:
        """Method -to be implemented- that preprocesses the data."""
