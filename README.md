# 🔮 Telco Customer Churn Predictor & ML Pipeline

An end-to-end production-ready Machine Learning pipeline to predict telecom customer churn. The project incorporates data quality validation, automated training, experiment tracking, unit testing, and model serving via a FastAPI and Gradio web interface, all containerized with Docker.

---

## 🚀 Key Features

* **Data Validation**: Enforces strict data quality and type checks at pipeline ingestion using `Pandera`.
* **Feature Engineering**: Implements categorical encoding and resolves dummy-variable multicollinearity (collapsing redundant "No internet service" indicators) based on Exploratory Data Analysis (EDA) findings.
* **Class Imbalance Handling**: Dynamically optimizes XGBoost's `scale_pos_weight` based on the dataset ratio to improve recall on the minority churn class.
* **Experiment Tracking**: Full lifecycle tracking of parameters, evaluation metrics (Accuracy, Recall, Precision, F1-Score, ROC AUC), and serialized models using `MLflow`.
* **API & Web Interface Serving**: A unified serving layer built on **FastAPI** (for programmatical REST requests) and **Gradio** (an interactive web UI for testing and demos).
* **Automated CI/CD**: A built-in GitHub Actions workflow that handles testing, training runs, and Docker image compilation on every commit.

---

## 🛠️ Tech Stack

* **Core Logic**: Python 3.11, Pandas, NumPy, Scikit-Learn
* **ML Model**: XGBoost Classifier
* **Tracking & Logging**: MLflow
* **Data Validation**: Pandera
* **API Serving & UI**: FastAPI, Uvicorn, Gradio, Pydantic, Joblib
* **Packaging & DevOps**: Docker, GitHub Actions, Pytest

---

## 📂 Project Structure

```
├── .github/workflows/
│   └── docker-build.yml     # CI/CD test, train & build automation workflow
├── artifacts/               # Serialized model and preprocessing assets (gitignored)
├── data/
│   └── Telco_Churn_Dataset.csv # Raw customer dataset
├── mlruns/                  # MLflow runs and logs directory (gitignored)
├── notebooks/
│   └── EDA.ipynb            # Exploratory Data Analysis notebook
├── scripts/
│   └── run_pipeline.py      # End-to-end ML training pipeline script
├── src/
│   ├── app/
│   │   └── main.py          # FastAPI serving API & Gradio UI
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py    # Standardized preprocessing functions
│   ├── features/
│   │   └── build_feature.py # Feature engineering, scaling & encoding
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── tune.py          # Hyperparameter tuning using Optuna
│   ├── serving/
│   │   └── inference.py     # Live inference engine
│   └── utils/
│       └── validate_data.py # Pandera schema definitions
├── tests/
│   └── test_pipeline.py     # Pytest unit testing suite
├── dockerfile               # Multi-stage optimized Docker deployment file
├── .dockerignore            # Docker context optimization exclusions
└── .gitignore               # Multi-layer git ignore configurations
```

---

## ⚙️ How to Setup & Run

### 1. Initialize the Environment
Activate your Python environment and install the required dependencies:
```bash
# Activate your local virtual environment (e.g. .venv-1)
.venv-1\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the ML Pipeline (Train & Track)
Trigger the training pipeline to validate raw data, build features, train the XGBoost classifier, log metrics to MLflow, and export prediction assets to `artifacts/`:
```bash
python scripts/run_pipeline.py
```

### 3. Run Unit Tests
Verify model stability, data loading, preprocessing logic, and model inference schemas:
```bash
python -m pytest
```
*Note: You can also execute the test suite directly as a standalone script: `python tests/test_pipeline.py`*

### 4. Start the Serving Web App
Launch the prediction API and Gradio UI locally:
```bash
python -m uvicorn src.app.main:app --port 8000
```
* Access the automated OpenAPI documentation at: `http://127.0.0.1:8000/docs`
* Access the interactive Churn Predictor GUI at: `http://127.0.0.1:8000/ui`

---

## 🐳 Docker Deployment

### Build the Image
To build the containerized serving layer (which bundles the frozen model weights automatically):
```bash
docker build -t telco-churn-api:latest .
```

### Run the Container
Spin up the serving container:
```bash
docker run -d -p 8000:8000 telco-churn-api:latest
```
Open your browser and navigate to `http://localhost:8000/ui` to run live inference inside Docker.

---

## 🙏 Acknowledgements

A special thanks to the following creators and resources whose educational guides and templates helped shape the EDA and ML pipeline architecture of this project:

* **Anas Riad**: For their excellent tutorial on [Machine Learning Pipelines](https://www.youtube.com/watch?v=luJ64trcCwc&t=2227s) and pipeline templates.
* **Siddhardhan**: For their insightful videos on [Exploratory Data Analysis (EDA)](https://youtu.be/qNglJgNOb7A?si=TbjhFmuiTim-tevD) and data preprocessing structures.
