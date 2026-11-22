import matplotlib
matplotlib.use('Agg')
import os
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import project modules
from utils import setup_logging, set_seed, download_dataset
from preprocessing import load_raw_data, clean_data, split_data
from feature_engineering import engineer_features_df, create_sliding_windows
from evaluate import evaluate_model_performance, compare_models
from explain import plot_tree_feature_importance, plot_permutation_importance_score, plot_shap_explanations

# PyTorch Deep Learning Classifier Architectures
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        # We take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, seq_len=10):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * (seq_len // 2), 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Conv1d expects input shape: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

class MaintenanceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_pytorch_model(model, train_loader, val_loader, epochs=15, lr=0.001, pos_weight=1.0, device="cpu"):
    """Train a PyTorch deep learning model and return the best state dictionary."""
    model = model.to(device)
    # Using BCEWithLogitsLoss with pos_weight to handle severe class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        
        logging.debug(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
    model.load_state_dict(best_model_state)
    return model

def prepare_data_pipelines(data_dir="data", models_dir="models"):
    """Download, clean, engineer features, split sequentially, and scale."""
    csv_path = download_dataset(data_dir)
    df = load_raw_data(csv_path)
    df = clean_data(df)
    
    # 1. Core Feature Engineering before scaling (physical parameters)
    df_feat = engineer_features_df(df, window_size=5)
    
    # 2. Sequential Splitting
    train_df, val_df, test_df = split_data(df_feat, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # 3. Categorical encoding
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    for d in [train_df, val_df, test_df]:
        d['type_encoded'] = d['type'].map(type_mapping)
        
    # 4. Scale numerical columns
    # We will scale all numeric features (raw and engineered)
    num_features = [
        'air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear',
        'temp_difference', 'power_index', 'overstrain_index',
        'air_temperature_roll_mean_5', 'air_temperature_roll_std_5',
        'process_temperature_roll_mean_5', 'process_temperature_roll_std_5',
        'rotational_speed_roll_mean_5', 'rotational_speed_roll_std_5',
        'torque_roll_mean_5', 'torque_roll_std_5',
        'tool_wear_roll_mean_5', 'tool_wear_roll_std_5',
        'rotational_speed_lag_1', 'rotational_speed_lag_2',
        'torque_lag_1', 'torque_lag_2',
        'air_temperature_lag_1', 'air_temperature_lag_2',
        'temp_difference_lag_1', 'temp_difference_lag_2'
    ]
    
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    test_df_scaled = test_df.copy()
    
    train_df_scaled[num_features] = scaler.fit_transform(train_df[num_features])
    val_df_scaled[num_features] = scaler.transform(val_df[num_features])
    test_df_scaled[num_features] = scaler.transform(test_df[num_features])
    
    # Also create scaled core columns separately for deep learning sequences
    dl_core_cols = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
    dl_scaler = StandardScaler()
    
    train_df_scaled[[f"{col}_scaled" for col in dl_core_cols]] = dl_scaler.fit_transform(train_df[dl_core_cols])
    val_df_scaled[[f"{col}_scaled" for col in dl_core_cols]] = dl_scaler.transform(val_df[dl_core_cols])
    test_df_scaled[[f"{col}_scaled" for col in dl_core_cols]] = dl_scaler.transform(test_df[dl_core_cols])
    
    # Save the scalers
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(dl_scaler, os.path.join(models_dir, "dl_scaler.joblib"))
    
    # Identify feature columns for traditional ML
    feature_cols = ['type_encoded'] + num_features
    
    return train_df_scaled, val_df_scaled, test_df_scaled, feature_cols, dl_core_cols

def run_main_training():
    """Main model training and hyperparameter optimization runner."""
    setup_logging()
    set_seed(42)
    
    models_dir = "models"
    figures_dir = "figures"
    
    # Prepare Data
    train_df, val_df, test_df, feature_cols, dl_core_cols = prepare_data_pipelines()
    
    # Prepare traditional ML variables
    X_train = train_df[feature_cols].values
    y_train = train_df['machine_failure'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['machine_failure'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['machine_failure'].values
    
    logging.info(f"Traditional ML features: {len(feature_cols)} features.")
    
    metrics_list = []
    
    # ------------------ Traditional ML Models ------------------
    
    # 1. Logistic Regression
    logging.info("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    y_prob = lr_model.predict_proba(X_test)[:, 1]
    metrics_list.append(evaluate_model_performance(y_test, y_pred, y_prob, "Logistic Regression", figures_dir))
    joblib.dump(lr_model, os.path.join(models_dir, "logistic_regression_best.joblib"))
    
    # 2. Decision Tree
    logging.info("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    y_prob = dt_model.predict_proba(X_test)[:, 1]
    metrics_list.append(evaluate_model_performance(y_test, y_pred, y_prob, "Decision Tree", figures_dir))
    joblib.dump(dt_model, os.path.join(models_dir, "decision_tree_best.joblib"))
    
    # 3. Random Forest (with Hyperparameter Tuning)
    logging.info("Tuning Random Forest...")
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    # Time-series cross validation splits
    tscv = TimeSeriesSplit(n_splits=3)
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=tscv, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    logging.info(f"Best RF Params: {rf_grid.best_params_}")
    
    y_pred = rf_best.predict(X_test)
    y_prob = rf_best.predict_proba(X_test)[:, 1]
    metrics_list.append(evaluate_model_performance(y_test, y_pred, y_prob, "Random Forest", figures_dir))
    joblib.dump(rf_best, os.path.join(models_dir, "random_forest_best.joblib"))
    
    # 4. XGBoost (with Hyperparameter Tuning)
    logging.info("Tuning XGBoost...")
    # Calculate scale_pos_weight for imbalance
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb_base = xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss')
    xgb_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1]
    }
    xgb_grid = GridSearchCV(xgb_base, xgb_param_grid, cv=tscv, scoring='f1', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    xgb_best = xgb_grid.best_estimator_
    logging.info(f"Best XGBoost Params: {xgb_grid.best_params_}")
    
    y_pred = xgb_best.predict(X_test)
    y_prob = xgb_best.predict_proba(X_test)[:, 1]
    metrics_list.append(evaluate_model_performance(y_test, y_pred, y_prob, "XGBoost", figures_dir))
    joblib.dump(xgb_best, os.path.join(models_dir, "xgboost_best.joblib"))
    
    # 5. LightGBM (with Hyperparameter Tuning)
    logging.info("Tuning LightGBM...")
    lgb_base = lgb.LGBMClassifier(scale_pos_weight=ratio, random_state=42, verbose=-1)
    lgb_param_grid = {
        'n_estimators': [50, 100],
        'num_leaves': [15, 31],
        'learning_rate': [0.05, 0.1]
    }
    lgb_grid = GridSearchCV(lgb_base, lgb_param_grid, cv=tscv, scoring='f1', n_jobs=-1)
    lgb_grid.fit(X_train, y_train)
    lgb_best = lgb_grid.best_estimator_
    logging.info(f"Best LightGBM Params: {lgb_grid.best_params_}")
    
    y_pred = lgb_best.predict(X_test)
    y_prob = lgb_best.predict_proba(X_test)[:, 1]
    metrics_list.append(evaluate_model_performance(y_test, y_pred, y_prob, "LightGBM", figures_dir))
    joblib.dump(lgb_best, os.path.join(models_dir, "lightgbm_best.joblib"))
    
    # 6. Support Vector Machine
    logging.info("Training Support Vector Machine...")
    svm_model = SVC(probability=True, class_weight='balanced', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]
    metrics_list.append(evaluate_model_performance(y_test, y_pred, y_prob, "SVM", figures_dir))
    joblib.dump(svm_model, os.path.join(models_dir, "svm_best.joblib"))
    
    # ------------------ Deep Learning Models ------------------
    logging.info("Preparing sliding windows for PyTorch Deep Learning models...")
    
    # Select variables for DL (using scaled raw variables + type_encoded)
    dl_features = ['type_encoded'] + [f"{col}_scaled" for col in dl_core_cols]
    
    train_dl_feats = train_df[dl_features].values
    train_dl_labels = train_df['machine_failure'].values
    
    val_dl_feats = val_df[dl_features].values
    val_dl_labels = val_df['machine_failure'].values
    
    test_dl_feats = test_df[dl_features].values
    test_dl_labels = test_df['machine_failure'].values
    
    # sliding window configuration
    window_size = 10
    
    X_train_seq, y_train_seq = create_sliding_windows(train_dl_feats, train_dl_labels, window_size)
    X_val_seq, y_val_seq = create_sliding_windows(val_dl_feats, val_dl_labels, window_size)
    X_test_seq, y_test_seq = create_sliding_windows(test_dl_feats, test_dl_labels, window_size)
    
    logging.info(f"DL sequences shape - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")
    
    # PyTorch DataLoaders
    train_dataset = MaintenanceDataset(X_train_seq, y_train_seq)
    val_dataset = MaintenanceDataset(X_val_seq, y_val_seq)
    test_dataset = MaintenanceDataset(X_test_seq, y_test_seq)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_seq.shape[2]
    # Class imbalance ratio for pos_weight
    pos_weight = ratio
    
    # 7. LSTM Model
    logging.info("Training LSTM...")
    lstm_model = LSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2)
    lstm_model = train_pytorch_model(lstm_model, train_loader, val_loader, epochs=15, pos_weight=pos_weight, device=device)
    torch.save(lstm_model.state_dict(), os.path.join(models_dir, "lstm_best.pt"))
    
    # Predict on test
    lstm_model.eval()
    y_test_prob = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = lstm_model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_test_prob.extend(probs)
    y_test_prob = np.array(y_test_prob).flatten()
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    metrics_list.append(evaluate_model_performance(y_test_seq, y_test_pred, y_test_prob, "LSTM", figures_dir))
    
    # 8. GRU Model
    logging.info("Training GRU...")
    gru_model = GRUClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2)
    gru_model = train_pytorch_model(gru_model, train_loader, val_loader, epochs=15, pos_weight=pos_weight, device=device)
    torch.save(gru_model.state_dict(), os.path.join(models_dir, "gru_best.pt"))
    
    # Predict on test
    gru_model.eval()
    y_test_prob = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = gru_model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_test_prob.extend(probs)
    y_test_prob = np.array(y_test_prob).flatten()
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    metrics_list.append(evaluate_model_performance(y_test_seq, y_test_pred, y_test_prob, "GRU", figures_dir))
    
    # 9. 1D CNN Model
    logging.info("Training 1D CNN...")
    cnn_model = CNN1DClassifier(input_dim=input_dim, seq_len=window_size)
    cnn_model = train_pytorch_model(cnn_model, train_loader, val_loader, epochs=15, pos_weight=pos_weight, device=device)
    torch.save(cnn_model.state_dict(), os.path.join(models_dir, "cnn1d_best.pt"))
    
    # Predict on test
    cnn_model.eval()
    y_test_prob = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = cnn_model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_test_prob.extend(probs)
    y_test_prob = np.array(y_test_prob).flatten()
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    metrics_list.append(evaluate_model_performance(y_test_seq, y_test_pred, y_test_prob, "1D CNN", figures_dir))
    
    # ------------------ Model Comparison & Final Best Selection ------------------
    df_comparison = compare_models(metrics_list, figures_dir)
    print("\n--- Final Performance Comparison ---")
    print(df_comparison.to_string())
    
    # Find best model based on F1-score
    best_model_name = df_comparison["f1_score"].idxmax()
    logging.info(f"The best overall performing model based on F1-score is: {best_model_name}")
    
    # Copy best model parameters to a general filename for standard predict loading
    # We will map the best model to best_model.joblib or best_model.pt
    best_model_key = best_model_name.lower().replace(" ", "_")
    if best_model_key in ["lstm", "gru", "1d_cnn"]:
        # Deep learning best model
        dl_map = {"lstm": "lstm_best.pt", "gru": "gru_best.pt", "1d_cnn": "cnn1d_best.pt"}
        src = os.path.join(models_dir, dl_map[best_model_key])
        dst = os.path.join(models_dir, "best_model.pt")
        import shutil
        shutil.copy(src, dst)
        logging.info(f"Saved PyTorch weights as {dst}")
    else:
        # Traditional best model
        src = os.path.join(models_dir, f"{best_model_key}_best.joblib")
        dst = os.path.join(models_dir, "best_model.joblib")
        import shutil
        shutil.copy(src, dst)
        logging.info(f"Saved joblib model as {dst}")
        
    # Write a quick model_meta.txt to record which model is the best
    with open(os.path.join(models_dir, "model_meta.txt"), "w") as f:
        f.write(f"best_model_name: {best_model_name}\n")
        f.write(f"best_model_key: {best_model_key}\n")
        
    # ------------------ Explainability with XGBoost / LightGBM / RF ------------------
    # Choose LightGBM or XGBoost or RF as the proxy tree-based model for explanations
    # Usually XGBoost/LightGBM has the best scores on this dataset and provides excellent importances
    expl_model = xgb_best
    expl_name = "XGBoost"
    
    logging.info(f"Generating feature importances and explanations using {expl_name}...")
    plot_tree_feature_importance(expl_model, feature_cols, expl_name, os.path.join(figures_dir, "xgboost_feature_importance.png"))
    plot_permutation_importance_score(expl_model, X_val, y_val, feature_cols, expl_name, os.path.join(figures_dir, "xgboost_permutation_importance.png"))
    plot_shap_explanations(expl_model, X_train, X_val, feature_cols, figures_dir)
    
    logging.info("Training script completed successfully!")

if __name__ == "__main__":
    run_main_training()
