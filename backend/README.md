# Manufacturing Quality Control & Predictive Maintenance using Time-Series Sensor Data

This is a complete end-to-end Machine Learning and Data Science project that predicts industrial equipment failures using time-series sensor data from the AI4I 2020 Predictive Maintenance Dataset.

The repository includes a comprehensive exploratory data analysis (EDA), custom feature engineering pipelines, multiple model evaluations (traditional classifiers vs. deep learning networks in PyTorch), hyperparameter tuning, model explainability (SHAP/Permutation Importance), and a local inference prediction pipeline.

---

## Project Structure

```
predictive_maintenance/
│
├── data/
│   └── ai4i2020.csv               # Raw dataset (unzipped programmatically)
│
├── notebooks/
│   ├── notebook_generator.py      # Programmatic generator for the jupyter notebook
│   └── predictive_maintenance.ipynb # Completed workflow notebook
│
├── src/
│   ├── preprocessing.py           # Data loading, cleaning, scaling, and splitting
│   ├── feature_engineering.py     # Rolling, lag, physics-based features, and sequences
│   ├── train.py                   # Tuning and training traditional ML + PyTorch deep learning
│   ├── evaluate.py                # Performance metric calculations & evaluation plots
│   ├── predict.py                 # Real-time inference script on new sensor measurements
│   ├── explain.py                 # Explainable AI (SHAP & Permutation Importances)
│   └── utils.py                   # Setup logger, seeding, and dataset downloader
│
├── models/
│   ├── scaler.joblib              # Saved fitted preprocessing scaler
│   ├── dl_scaler.joblib           # Scaler for Deep Learning raw sequences
│   ├── best_model.joblib          # Copied best model artifact (XGBoost)
│   ├── xgboost_best.joblib        # Saved tuned XGBoost model
│   ├── lstm_best.pt               # Saved trained PyTorch LSTM weights
│   └── ...                        # Other models (GRU, 1D CNN, LightGBM, Random Forest, SVM)
│
├── reports/
│   └── final_report.md            # Detailed methodology, results, and discussion
│
├── figures/
│   └── (saved curves, heatmaps, importance plots, and comparison CSV)
│
├── requirements.txt               # Package dependencies list
└── README.md                      # Setup and usage guide
```

---

## Installation & Setup

1. **Python Version**: Recommended Python 3.10 - 3.13.
2. **Install Dependencies**:
   Navigate to the project directory and install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Project

### Step 1: Run Training & Evaluation
Run the training script to download the dataset, preprocess it, engineer features, train/tune all models, evaluate them on the test partition, and generate explainability plots:
```bash
python src/train.py
```
This script will:
- Save all models to the `models/` directory.
- Save metrics and plots (ROC curves, PR curves, Confusion Matrices, and SHAP plots) to the `figures/` directory.
- Output a comparison table of all model performances in the console.

### Step 2: Generate the Jupyter Notebook
Run the notebook generator program to create the interactive `notebooks/predictive_maintenance.ipynb` containing the complete narrative workflow:
```bash
python notebooks/notebook_generator.py
```

### Step 3: Run Inference (Predictions)
Use the `predict.py` command-line script to predict the failure probability on new sensor reading sequences. Pass the `--model` argument to load any specific model (e.g. `xgboost`, `lightgbm`, `lstm`, `gru`):
```bash
python src/predict.py --model xgboost
```

---

## Model Comparison Summary

The models were evaluated on the final 1,500 sequential rows of the dataset.

*   **Best Overall Classifier**: **XGBoost (Tuned)** with an **F1-Score of 0.8077**, Precision of 91.30%, and Recall of 72.41%.
*   **Highest Recall Classifier**: **LSTM (PyTorch)** with a **Recall of 78.57%**. 
    *   *Note*: High Recall is critical in industrial maintenance as missing a failure (False Negative) is significantly more expensive than inspecting a functioning machine (False Positive).

Refer to [reports/final_report.md](reports/final_report.md) for the detailed explanation and analysis.
