# Customer Support Ticket Intelligence

A machine learning pipeline and interactive intelligence system for classifying, prioritizing, analyzing, and routing customer support tickets.

## 📌 Features

- **Ticket Categorization & Routing**: Automatically assign incoming support tickets to appropriate departments (e.g., Billing, Technical Support, Account Management).
- **Urgent Ticket Detection & Escalation**: Flag high-priority and urgent issues based on sentiment and urgency analysis.
- **Feature Extraction & NLP Pipeline**: Preprocess unstructured text, compute embeddings, and extract key metadata.
- **API Endpoint (FastAPI)**: RESTful API for real-time ticket inference and batch processing.
- **Interactive Dashboard (Streamlit)**: Live analytics interface for support leads and agents to view ticket trends, performance metrics, and trigger automated routing.

---

## 📁 Repository Structure

```
customer-support-ticket-intelligence/
├── data/
│   ├── raw/                # Raw, unmodified support ticket datasets
│   └── processed/          # Cleaned, tokenized, and preprocessed datasets
├── notebooks/              # Jupyter notebooks for EDA, prototyping, and analysis
├── src/                    # Core Python source package
│   ├── data/               # Data loading, cleaning, and ingestion scripts
│   ├── features/           # Feature extraction, embeddings, and text preprocessing
│   ├── models/             # Model architectures, training scripts, and pipelines
│   ├── evaluation/         # Evaluation metrics, performance plots, and benchmarking
│   └── inference/          # Prediction scripts and inference engine
├── api/                    # FastAPI web service for real-time predictions
├── dashboard/              # Streamlit interactive dashboard for ticket analytics
├── tests/                  # Unit and integration test suite
├── models/                 # Serialized model artifacts and checkpoints
├── configs/                # Project configuration files (YAML/JSON)
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup

Ensure you have Python 3.9+ installed.

```bash
# Clone repository
git clone https://github.com/your-org/customer-support-ticket-intelligence.git
cd customer-support-ticket-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Review and customize configuration settings in `configs/config.yaml`.

---

## 💻 Running the Services

### Start the REST API

Launch the FastAPI server for ticket inference:

```bash
uvicorn api.main:app --reload --port 8000
```

Access API Documentation at `http://localhost:8000/docs`.

### Start the Dashboard

Launch the Streamlit analytics interface:

```bash
streamlit run dashboard/app.py
```

---

## 🧪 Running Tests

Execute the unit test suite with `pytest`:

```bash
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
