from dataclasses import dataclass

@dataclass
class Config:
    tracking_uri = r"sqlite:///mlruns.db"
    experiment_name = r"Iris-Flower-Classification-Experiment"
    description_name = "Logistic Regression Classification run for Iris Flower data set"
