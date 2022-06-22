from abc import ABC, abstractmethod


class DataLoader(ABC):
    @abstractmethod
    def load_data(self) -> None:
        """Method -to be implemented- that loads the data."""
