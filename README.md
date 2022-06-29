# Getting started
Setting up a virtual environment to separate your projects (pip) dependencies. 

```bash
py -m venv venv
```

Active your virtual environment

```bash
source venv/Scripts/active # Windows
source venv/bin/activate # Unix
```

Install the projects requirements.

```bash
pip install -r requirements.txt
```

Install the model package (src) and the training template (training_template)

```bash
pip install -e src/
pip install -e training_template/
```

Running the Tracking server for the dashboard on https://127.0.0.1:5000

```bash
./start_server.sh
```

# Training Template
The main focus here is to standardize the Training of Machine Learning models.
This is done by providing Abstract Base Classes for components of the Model Training Pipeline.

# src folder for our Iris flower classification model
The specific Machine Learning model implementation. 
The steps are the implementations of the Abstract Base Classes as defined in the training template.