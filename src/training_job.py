import pickle
import tempfile
from pathlib import Path

import mlflow
from config import Config
from data_loading import IrisDataLoader
from data_monitoring import IrisDataMonitor
from data_preprocessing import IrisDataPreprocessor
from model_evaluating import IrisModelEvaluator
from model_training import IrisModelTrainer
from training_template.data_loader import DataLoader
from training_template.data_monitor import DataMonitor
from training_template.data_preprocessor import DataPreprocessor
from training_template.job import Job
from training_template.model_evaluator import ModelEvaluator
from training_template.model_trainer import ModelTrainer


class TrainingJob(Job):  # Supervised
    def __init__(
        self,
        config: Config,
        data_loader: DataLoader,
        data_preprocessor: DataPreprocessor,
        model_trainer: ModelTrainer,
        model_evaluator: ModelEvaluator,
        data_monitor: DataMonitor,
    ):

        """Dependency Injection for Steps in Model Training Pipeline."""
        self.config = config
        self.data_loader = data_loader
        self.data_preprocessor = data_preprocessor
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator
        self.data_monitor = data_monitor

        self.data = None
        self.data_evaluation = None
        self.preprocessed_data = None
        self.train_data = None
        self.model = None
        self.predictions = None
        self.evaluations = None
        self.model_evaluation = None
        self.predictions = None

    def launch(self):
        print("Launching Job")
        # Set tracking URI
        mlflow.set_tracking_uri(self.config.tracking_uri)

        # Set experiment name
        mlflow.set_experiment(self.config.experiment_name)

        # Start experiment run
        with mlflow.start_run(description=self.config.description_name):

            print("Loading data")
            self.load_data()
            for key in self.data.keys():
                shape = self.data[key].shape
                print(f"{key} has shape {self.data[key].shape}")
                key_name = key + "_" + "shape"
                mlflow.log_param(key_name, shape)

            print("Data monitoring")

            print("Preprocessing data")
            self.preprocess_data()
            for key in self.preprocessed_data.keys():
                shape = self.preprocessed_data[key].shape
                print(f"{key} has shape {shape}")
                key_name = key + "_" + "shape"
                mlflow.log_param(key_name, shape)

            # Subset train data from self.preprocessed_data

            print("Training model")
            self.train_model()

            with tempfile.TemporaryDirectory() as tempdir:
                model_path = Path(tempdir) / "model.pickle"
                with open(model_path, "wb") as f:
                    pickle.dump(self.model, f)
                    mlflow.log_artifact(model_path)

            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="sklearn-model",
                registered_model_name="sk-learn-log-regression-model",
            )

            print(self.model.predict(self.preprocessed_data["X_train"]))

            print("Evaluating model")
            self.evaluate_model()
            for key in self.model_evaluation.keys():
                metric = self.model_evaluation[key]
                print(f"Metric {key} has value {metric:.2f}")
                mlflow.log_param(key, metric)

    def load_data(self):
        # Check whether data is already available
        if not self.data:
            self.data = self.data_loader.load_data()

    def preprocess_data(self):
        self.load_data()
        # Check whether preprocessed data is already available
        if self.preprocessed_data is None:
            self.preprocessed_data = self.data_preprocessor.preprocess_data(data=self.data)

    def train_model(self):
        self.preprocess_data()
        if self.train_data is None:
            self.train_data = {
                key if "train" in key else ...: self.preprocessed_data[key] for key in self.preprocessed_data.keys()
            }
        if self.model is None:
            self.model = self.model_trainer.train_model(data=self.train_data)

    def evaluate_model(self):
        self.train_model()

        # Use model to generate predictions
        if self.predictions is None:
            self.predictions = self.model.predict(self.preprocessed_data["X_test"])

        # Use predictions to evaluate model
        if self.model_evaluation is None:
            self.model_evaluation = self.model_evaluator.evaluate_model(
                predictions=self.predictions, y_test=self.preprocessed_data["y_test"]
            )


if __name__ == "__main__":
    # Initialize config
    config = Config()

    # Initialize specific DataLoader: IrisDataLoader
    data_loader = IrisDataLoader()

    # Initialize specific DataMonitor: IrisDataMonitor
    data_monitor = IrisDataMonitor()

    # Initialize specific DataPreprocessor: IrisDataPreprocessor
    data_preprocessor = IrisDataPreprocessor()

    # Initialize specific ModelTrainer: IrisModelTrainer
    model_trainer = IrisModelTrainer()

    # Initialize specific ModelEvaluator: IrisModelEvaluator
    model_evaluator = IrisModelEvaluator()

    # Use Dependency Injection to set up Training Job with the specific implementation
    training_job = TrainingJob(
        config=config,
        data_loader=data_loader,
        data_preprocessor=data_preprocessor,
        model_trainer=model_trainer,
        model_evaluator=model_evaluator,
        data_monitor=data_monitor,
    )

    # Execute the Training Job with launch method
    training_job.launch()
