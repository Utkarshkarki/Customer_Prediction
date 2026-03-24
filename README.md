# 🧠 Customer Prediction — ML Pipeline

> An end-to-end Machine Learning pipeline to predict customer behavior, served via a **FastAPI** web application and backed by **MongoDB** for data storage and **AWS S3** for model artifact management.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
  - [Local Development](#local-development)
  - [Docker](#docker)
- [ML Pipeline Stages](#ml-pipeline-stages)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [License](#license)

---

## 📖 Overview

This project implements a production-ready **Customer Prediction System** that automates the complete machine learning lifecycle:

1. **Data Ingestion** — Pulls raw customer data from a MongoDB database.
2. **Data Validation** — Validates schema, detects drift, and ensures data quality.
3. **Data Transformation** — Applies feature engineering and preprocessing.
4. **Model Training** — Trains classification models (XGBoost, CatBoost, scikit-learn).
5. **Model Evaluation** — Evaluates model performance and compares against the production baseline.
6. **Model Pusher** — Pushes the best-performing model to AWS S3.
7. **Prediction API** — Serves real-time predictions via a FastAPI REST interface.

---

## 🏗️ Architecture

```
MongoDB (Raw Data)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
│                                                         │
│  Data Ingestion → Validation → Transformation           │
│        → Model Training → Evaluation → Model Pusher     │
└─────────────────────────────────────────────────────────┘
       │
       ▼
   AWS S3 (Model Artifacts)
       │
       ▼
┌─────────────────┐
│  FastAPI Server │ ◄── REST API (Prediction Pipeline)
└─────────────────┘
```

---

## 📁 Project Structure

```
ProjectX/
├── Customer_Prediction/          # Core ML package
│   ├── components/               # Pipeline stage implementations
│   │   ├── data_ingestion.py     # Pulls data from MongoDB
│   │   ├── data_validation.py    # Schema & drift validation
│   │   ├── data_transformation.py# Feature engineering & preprocessing
│   │   ├── model_trainer.py      # Model training logic
│   │   ├── model_evaluation.py   # Performance evaluation
│   │   └── model_pusher.py       # Pushes model artifacts to AWS S3
│   ├── configuration/            # Component configurations
│   ├── constants/                # Project-wide constants
│   ├── entity/                   # Data classes / config entities
│   ├── exception/                # Custom exception handling
│   ├── logger/                   # Logging setup
│   ├── pipline/                  # Pipeline orchestrators
│   │   ├── training_pipeline.py  # End-to-end training orchestration
│   │   └── prediction_pipeline.py# Inference orchestration
│   └── utils/                    # Shared utility functions
│
├── config/
│   ├── model.yaml                # Model hyperparameter configuration
│   └── schema.yaml               # Dataset schema definition
│
├── Notebook/                     # Jupyter notebooks for EDA & experiments
├── app.py                        # FastAPI application entry point
├── demo.py                       # Quick demo / sanity check script
├── template.py                   # Project scaffolding script
├── setup.py                      # Package setup
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container definition
├── .dockerignore                 # Docker build exclusions
├── .gitignore                    # Git exclusions
└── .env                          # Environment variables (not committed)
```

---

## 🛠️ Tech Stack

| Category            | Tools / Libraries                              |
|---------------------|------------------------------------------------|
| **Language**        | Python 3.x                                     |
| **Web Framework**   | FastAPI, Uvicorn                               |
| **ML Libraries**    | scikit-learn, XGBoost, CatBoost, imbalanced-learn |
| **Data Processing** | Pandas, NumPy, SciPy                           |
| **Visualization**   | Matplotlib, Seaborn, Plotly                    |
| **Database**        | MongoDB (via PyMongo)                          |
| **Cloud Storage**   | AWS S3 (via Boto3)                             |
| **Monitoring**      | Evidently (Data Drift Detection)               |
| **Serialization**   | Dill, PyYAML                                   |
| **Containerization**| Docker                                         |
| **Environment**     | python-dotenv, pydantic                        |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python **3.8+**
- [pip](https://pip.pypa.io/en/stable/)
- [MongoDB](https://www.mongodb.com/) (local or Atlas)
- [Docker](https://www.docker.com/) *(optional, for containerized run)*
- AWS Account with S3 access *(for model pushing)*

---

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Utkarshkarki/ProjectX.git
   cd ProjectX
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Environment Variables

Create a `.env` file in the project root and configure the following variables:

```env
# MongoDB
MONGODB_URL=mongodb+srv://<username>:<password>@cluster.mongodb.net/<dbname>

# AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your_s3_bucket_name
```

> ⚠️ **Never commit your `.env` file.** It is already listed in `.gitignore`.

---

## ▶️ Running the Application

### Local Development

**Trigger the Training Pipeline:**
```bash
python demo.py
```

**Start the FastAPI Server:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Then open your browser at: [http://localhost:8080](http://localhost:8080)

---

### Docker

**Build the Docker image:**
```bash
docker build -t customer-prediction .
```

**Run the container:**
```bash
docker run -p 8080:8080 --env-file .env customer-prediction
```

---

## 🔄 ML Pipeline Stages

| Stage | Description |
|-------|-------------|
| **Data Ingestion** | Connects to MongoDB, exports data, splits into train/test sets |
| **Data Validation** | Validates column schema, checks for missing values, detects data drift using Evidently |
| **Data Transformation** | Applies imputation, scaling, encoding, and handles class imbalance (SMOTE via `imblearn`) |
| **Model Training** | Trains multiple classifiers; selects best model via cross-validation |
| **Model Evaluation** | Evaluates against a held-out test set; compares with production model |
| **Model Pusher** | Serializes and uploads the best model to AWS S3 for serving |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check / welcome page |
| `GET`  | `/train` | Triggers the full training pipeline |
| `POST` | `/predict` | Returns prediction for submitted customer data |

---

## ⚙️ Configuration

### `config/model.yaml`
Defines model hyperparameters and the list of candidate estimators used during training.

### `config/schema.yaml`
Defines the expected feature schema (column names, data types) for validation during data ingestion.

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/Utkarshkarki">Utkarsh Karki</a>
</p>