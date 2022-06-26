from abc import ABC, abstractmethod


class DataMonitor(ABC):
    @abstractmethod
    def monitor_data(self) -> None:
        """Method -to be implemented- that monitors the data."""
