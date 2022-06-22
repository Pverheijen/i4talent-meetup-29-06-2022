from abc import ABC, abstractmethod


class Job(ABC):
    @abstractmethod
    def launch(self) -> None:
        """The -to be implemented- launch method that should be implemented kicks off the job."""
