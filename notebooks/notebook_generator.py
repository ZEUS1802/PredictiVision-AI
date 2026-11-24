import os
import nbformat as nbf

def generate_notebook():
    nb = nbf.v4.new_notebook()
    
    # Cells list
    cells = []
    
    # 1. Title
    cells.append(nbf.v4.new_markdown_cell(
        "# Manufacturing Quality Control & Predictive Maintenance using Time-Series Sensor Data\n\n"
        "This Jupyter Notebook demonstrates the complete end-to-end Machine Learning pipeline to predict "
        "industrial equipment failures using time-series sensor data from the AI4I 2020 Predictive Maintenance Dataset.\n\n"
        "### Project Structure\n"
        "* **1. Setup & Imports**: Logging configuration, seed initialization, and importing custom modules.\n"
        "* **2. Exploratory Data Analysis (EDA)**: Statistical analysis, missing values, duplicates, outliers, correlations, and class distribution.\n"
        "* **3. Data Preprocessing & Split**: Sequential splitting and scaling.\n"
        "* **4. Feature Engineering**: Physics-based indicators, rolling statistics, lag features, and sliding windows.\n"
        "* **5. Machine Learning Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, SVM, and deep learning LSTM, GRU, 1D CNN.\n"
        "* **6. Model Comparison & Selection**: Comparing models by key classification metrics.\n"
        "* **7. Explainable AI (XAI)**: SHAP analysis, traditional feature importances, and permutation importances.\n"
        "* **8. Live Predictive Maintenance Predictor**: Simulating predictions on a new stream of sensor measurements."
    ))
    
    # 2. Imports and Setup
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "import os\n"
        "# Add src to python path\n"
        "sys.path.append(os.path.abspath('../src'))\n\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import joblib\n"
        "import torch\n\n"
        "from utils import setup_logging, set_seed, download_dataset\n"
        "from preprocessing import load_raw_data, clean_data, split_data, detect_outliers, preprocess_pipeline\n"
        "from feature_engineering import engineer_features_df, create_sliding_windows\n"
        "from evaluate import evaluate_model_performance, compare_models\n"
        "from explain import plot_tree_feature_importance, plot_permutation_importance_score, plot_shap_explanations\n"
        "from predict import predict_single_step\n\n"
        "# Setup environment\n"
        "setup_logging(log_file='../predictive_maintenance_nb.log')\n"
        "set_seed(42)\n"
        "print('Imports and environment setup complete.')"
    ))
    
    # 3. Load data
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load the AI4I 2020 Predictive Maintenance Dataset"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "csv_path = download_dataset(data_dir='../data')\n"
        "df = load_raw_data(csv_path)\n"
        "df = clean_data(df)\n"
        "df.head()"
    ))
    
    # 4. EDA Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Exploratory Data Analysis (EDA)\n\n"
        "Let's look at numerical distributions, the failure labels class distribution, "
        "and correlations between variables."
    ))
    
    # 5. EDA Code
    cells.append(nbf.v4.new_code_cell(
        "# Statistical Summary\n"
        "print('--- Statistical Summary ---')\n"
        "display(df.describe())\n\n"
        "# Class Distribution\n"
        "failure_counts = df['machine_failure'].value_counts()\n"
        "failure_pct = df['machine_failure'].value_counts(normalize=True) * 100\n"
        "print(f'Normal operations (0): {failure_counts[0]} ({failure_pct[0]:.2f}%)')\n"
        "print(f'Machine Failures (1):  {failure_counts[1]} ({failure_pct[1]:.2f}%)')\n\n"
        "# Plot Class Distribution\n"
        "plt.figure(figsize=(6, 4))\n"
        "sns.countplot(x='machine_failure', data=df, palette='Set2')\n"
        "plt.title('Machine Failure Label Distribution')\n"
        "plt.xlabel('Label (0: Normal, 1: Failed)')\n"
        "plt.ylabel('Count')\n"
        "plt.tight_layout()\n"
        "plt.savefig('../figures/class_distribution.png', dpi=300)\n"
        "plt.show()"
    ))
    
    # 6. EDA Outliers and Correlations
    cells.append(nbf.v4.new_code_cell(
        "# Outlier detection\n"
        "numeric_cols = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']\n"
        "detect_outliers(df, numeric_cols)\n\n"
        "# Correlation Matrix Heatmap\n"
        "plt.figure(figsize=(10, 8))\n"
        "corr = df[numeric_cols + ['machine_failure']].corr()\n"
        "sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5)\n"
        "plt.title('Correlation Matrix of Core Sensor Signals')\n"
        "plt.tight_layout()\n"
        "plt.savefig('../figures/correlation_heatmap.png', dpi=300)\n"
        "plt.show()"
    ))
    
    # 7. Time series trend analysis
    cells.append(nbf.v4.new_code_cell(
        "# Sensor readings leading to failure (sensor trends)\n"
        "failures_indices = df[df['machine_failure'] == 1].index\n"
        "if len(failures_indices) > 0:\n"
        "    idx = failures_indices[10] # look around the 10th failure event\n"
        "    # Plot window of 50 steps before/after failure\n"
        "    window_df = df.iloc[max(0, idx-40):min(len(df), idx+10)]\n"
        "    \n"
        "    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)\n"
        "    axes[0].plot(window_df.index, window_df['rotational_speed'], color='blue', label='Rotational Speed')\n"
        "    axes[0].set_ylabel('Speed (rpm)')\n"
        "    axes[0].legend(loc='upper left')\n"
        "    \n"
        "    axes[1].plot(window_df.index, window_df['torque'], color='red', label='Torque')\n"
        "    axes[1].set_ylabel('Torque (Nm)')\n"
        "    axes[1].legend(loc='upper left')\n"
        "    \n"
        "    axes[2].plot(window_df.index, window_df['tool_wear'], color='green', label='Tool Wear')\n"
        "    axes[2].set_ylabel('Tool Wear (min)')\n"
        "    axes[2].legend(loc='upper left')\n"
        "    \n"
        "    # Highlight the failure point\n"
        "    for ax in axes:\n"
        "        ax.axvline(x=idx, color='black', linestyle='--', lw=2, label='Failure Event')\n"
        "        \n"
        "    plt.suptitle('Sensor Readings and Failure progression', fontsize=14)\n"
        "    plt.tight_layout()\n"
        "    plt.savefig('../figures/sensor_failure_progression.png', dpi=300)\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No failure samples found for time-series analysis.')"
    ))
    
    # 8. Preprocessing Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Data Preprocessing & Feature Engineering\n\n"
        "We split our data chronologically into train (70%), validation (15%), and test (15%). "
        "We then perform feature engineering and scale numerical attributes. For Deep Learning models, we generate sliding window sequences."
    ))
    
    # 9. Preprocessing Code
    cells.append(nbf.v4.new_code_cell(
        "# Feature Engineering\n"
        "df_feat = engineer_features_df(df, window_size=5)\n\n"
        "# Split data sequentially\n"
        "train_df, val_df, test_df = split_data(df_feat, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)\n\n"
        "# Fit scaler and process\n"
        "type_mapping = {'L': 0, 'M': 1, 'H': 2}\n"
        "for d in [train_df, val_df, test_df]:\n"
        "    d['type_encoded'] = d['type'].map(type_mapping)\n\n"
        "num_features = [\n"
        "    'air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear',\n"
        "    'temp_difference', 'power_index', 'overstrain_index',\n"
        "    'air_temperature_roll_mean_5', 'air_temperature_roll_std_5',\n"
        "    'process_temperature_roll_mean_5', 'process_temperature_roll_std_5',\n"
        "    'rotational_speed_roll_mean_5', 'rotational_speed_roll_std_5',\n"
        "    'torque_roll_mean_5', 'torque_roll_std_5',\n"
        "    'tool_wear_roll_mean_5', 'tool_wear_roll_std_5',\n"
        "    'rotational_speed_lag_1', 'rotational_speed_lag_2',\n"
        "    'torque_lag_1', 'torque_lag_2',\n"
        "    'air_temperature_lag_1', 'air_temperature_lag_2',\n"
        "    'temp_difference_lag_1', 'temp_difference_lag_2'\n"
        "]\n\n"
        "scaler = StandardScaler()\n"
        "train_df_scaled = train_df.copy()\n"
        "val_df_scaled = val_df.copy()\n"
        "test_df_scaled = test_df.copy()\n\n"
        "train_df_scaled[num_features] = scaler.fit_transform(train_df[num_features])\n"
        "val_df_scaled[num_features] = scaler.transform(val_df[num_features])\n"
        "test_df_scaled[num_features] = scaler.transform(test_df[num_features])\n\n"
        "feature_cols = ['type_encoded'] + num_features\n"
        "X_train = train_df_scaled[feature_cols].values\n"
        "y_train = train_df_scaled['machine_failure'].values\n"
        "X_test = test_df_scaled[feature_cols].values\n"
        "y_test = test_df_scaled['machine_failure'].values\n"
        "print('Preprocessed data shapes:', X_train.shape, X_test.shape)"
    ))
    
    # 10. DL Preparation
    cells.append(nbf.v4.new_code_cell(
        "# Preparing Sliding Window sequences for Deep Learning models\n"
        "dl_core_cols = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']\n"
        "dl_scaler = StandardScaler()\n"
        "train_df_scaled[[f'{col}_scaled' for col in dl_core_cols]] = dl_scaler.fit_transform(train_df[dl_core_cols])\n"
        "val_df_scaled[[f'{col}_scaled' for col in dl_core_cols]] = dl_scaler.transform(val_df[dl_core_cols])\n"
        "test_df_scaled[[f'{col}_scaled' for col in dl_core_cols]] = dl_scaler.transform(test_df[dl_core_cols])\n\n"
        "dl_features = ['type_encoded'] + [f'{col}_scaled' for col in dl_core_cols]\n"
        "train_dl_feats = train_df_scaled[dl_features].values\n"
        "train_dl_labels = train_df_scaled['machine_failure'].values\n"
        "test_dl_feats = test_df_scaled[dl_features].values\n"
        "test_dl_labels = test_df_scaled['machine_failure'].values\n\n"
        "window_size = 10\n"
        "X_train_seq, y_train_seq = create_sliding_windows(train_dl_feats, train_dl_labels, window_size)\n"
        "X_test_seq, y_test_seq = create_sliding_windows(test_dl_feats, test_dl_labels, window_size)\n"
        "print('DL sequences shapes - Train:', X_train_seq.shape, 'Test:', X_test_seq.shape)"
    ))
    
    # 11. Model Training Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Run Model Training & Evaluation\n\n"
        "Rather than training from scratch in this notebook (which takes time), we can load the models trained "
        "by our scripts, predict on the test set, and compare their performance curves."
    ))
    
    # 12. Load and evaluate models
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import glob\n\n"
        "print('Loading pre-trained models from /models...')\n"
        "models_list = ['logistic_regression', 'decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'svm']\n"
        "metrics_results = []\n\n"
        "for m in models_list:\n"
        "    model_file = f'../models/{m}_best.joblib'\n"
        "    if os.path.exists(model_file):\n"
        "        model = joblib.load(model_file)\n"
        "        y_pred = model.predict(X_test)\n"
        "        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)\n"
        "        met = evaluate_model_performance(y_test, y_pred, y_prob, m.replace('_', ' ').title(), '../figures')\n"
        "        metrics_results.append(met)\n"
        "    else:\n"
        "        print(f'Model file {model_file} not found. Make sure train.py has run successfully.')"
    ))
    
    # 13. Evaluate Deep Learning
    cells.append(nbf.v4.new_code_cell(
        "# Load Deep Learning PyTorch Models and Predict\n"
        "from train import LSTMClassifier, GRUClassifier, CNN1DClassifier\n\n"
        "dl_models = {'lstm': LSTMClassifier(6), 'gru': GRUClassifier(6), 'cnn1d': CNN1DClassifier(6, window_size)}\n"
        "for name, model in dl_models.items():\n"
        "    weights_path = f'../models/{name}_best.pt'\n"
        "    if os.path.exists(weights_path):\n"
        "        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))\n"
        "        model.eval()\n"
        "        with torch.no_grad():\n"
        "            inputs_tensor = torch.tensor(X_test_seq, dtype=torch.float32)\n"
        "            outputs = model(inputs_tensor)\n"
        "            y_prob = torch.sigmoid(outputs).numpy().flatten()\n"
        "            y_pred = (y_prob >= 0.5).astype(int)\n"
        "            met = evaluate_model_performance(y_test_seq, y_pred, y_prob, name.upper().replace('CNN1D', '1D CNN'), '../figures')\n"
        "            metrics_results.append(met)\n"
        "    else:\n"
        "        print(f'DL Weights {weights_path} not found.')"
    ))
    
    # 14. Performance Comparison Plot
    cells.append(nbf.v4.new_code_cell(
        "# Compare Models\n"
        "if len(metrics_results) > 0:\n"
        "    df_comparison = compare_models(metrics_results, '../figures')\n"
        "    display(df_comparison)\n"
        "else:\n"
        "    print('No metrics results collected to compare.')"
    ))
    
    # 15. Explainability Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Explainable AI (XAI)\n\n"
        "Let's look at which sensor signals are driving the predictions using the SHAP plots and feature importance charts "
        "generated by our training scripts."
    ))
    
    # 16. Show plots
    cells.append(nbf.v4.new_code_cell(
        "from IPython.display import Image, display\n\n"
        "print('Feature Importance Plot from XGBoost:')\n"
        "if os.path.exists('../figures/xgboost_feature_importance.png'):\n"
        "    display(Image(filename='../figures/xgboost_feature_importance.png', width=600))\n\n"
        "print('SHAP Summary Plot:')\n"
        "if os.path.exists('../figures/shap_summary_plot.png'):\n"
        "    display(Image(filename='../figures/shap_summary_plot.png', width=600))"
    ))
    
    # 17. Live Predict Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Live Inference Demonstration\n\n"
        "Using the prediction utility, we simulate a stream of 10 new readings from a machine, passing them into "
        "our prediction system to compute real-time failure probabilities."
    ))
    
    # 18. Live Predict Code
    cells.append(nbf.v4.new_code_cell(
        "from predict import predict_single_step\n\n"
        "# Generate 10 consecutive readings\n"
        "sample_history = pd.DataFrame({\n"
        "    'type': ['L'] * 10,\n"
        "    'air_temperature': [298.0 + 0.1 * i for i in range(10)],\n"
        "    'process_temperature': [308.0 + 0.1 * i for i in range(10)],\n"
        "    'rotational_speed': [1450 - 15 * i for i in range(10)],\n"
        "    'torque': [45.0 + 1.2 * i for i in range(10)],\n"
        "    'tool_wear': [120 + 2 * i for i in range(10)]\n"
        "})\n\n"
        "# Predict using best model\n"
        "# Check model_meta.txt for best model\n"
        "model_type = 'xgboost'\n"
        "if os.path.exists('../models/model_meta.txt'):\n"
        "    with open('../models/model_meta.txt', 'r') as f:\n"
        "        lines = f.readlines()\n"
        "        for line in lines:\n"
        "            if 'best_model_key' in line:\n"
        "                model_type = line.split(':')[1].strip()\n"
        "                break\n\n"
        "print(f'Running live prediction using the best model: {model_type}')\n"
        "pred, prob = predict_single_step(sample_history, models_dir='../models', model_type=model_type)\n"
        "print('--- Inference Results ---')\n"
        "print(f'Machine Failure Predicted: {\"YES\" if pred == 1 else \"NO\"}')\n"
        "print(f'Failure Probability: {prob:.4%}')"
    ))
    
    # Assemble Notebook
    nb.cells = cells
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(script_dir, "predictive_maintenance.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Jupyter Notebook successfully created at {notebook_path}")

if __name__ == '__main__':
    generate_notebook()
