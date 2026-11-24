import os
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PredictiveMaintenancePredictor")

# Load PyTorch DL models locally if needed
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Take the output of the last time step
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
        # x shape: (batch, seq_len, input_dim)
        # Conv1d expects (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

def preprocess_new_data(history_df, scaler, dl_scaler=None, window_size=5):
    """
    Given a history DataFrame, engineer features and scale numerical values.
    Returns the latest row (as a DataFrame) for traditional ML.
    """
    df = history_df.copy()
    
    # 1. Physics features
    df['temp_difference'] = df['process_temperature'] - df['air_temperature']
    df['power_index'] = df['rotational_speed'] * df['torque']
    df['overstrain_index'] = df['tool_wear'] * df['torque']
    
    # 2. Rolling stats
    cols_to_roll = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
    for col in cols_to_roll:
        df[f'{col}_roll_mean_{window_size}'] = df[col].rolling(window=window_size, min_periods=1).mean()
        df[f'{col}_roll_std_{window_size}'] = df[col].rolling(window=window_size, min_periods=1).std().fillna(0.0)
        
    # 3. Lags
    cols_to_lag = ['rotational_speed', 'torque', 'air_temperature', 'temp_difference']
    for col in cols_to_lag:
        df[f'{col}_lag_1'] = df[col].shift(1).bfill()
        df[f'{col}_lag_2'] = df[col].shift(2).bfill()
        
    # 4. Encoding
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    df['type_encoded'] = df['type'].map(type_mapping)
    
    # 5. Scaling in-place to match train.py exactly
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
    
    df[num_features] = scaler.transform(df[num_features])
    
    # 6. DL scaling (with _scaled suffix) if dl_scaler is provided
    if dl_scaler is not None:
        dl_core_cols = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
        df[[f"{col}_scaled" for col in dl_core_cols]] = dl_scaler.transform(history_df[dl_core_cols])
        
    feature_cols = ['type_encoded'] + num_features
    X = df[feature_cols]
    
    # Return the latest row as a DataFrame to preserve feature names
    return X.iloc[[-1]], df

def predict_single_step(history_df, models_dir="models", model_type="xgboost", window_size=5):
    """
    Predict machine failure for the latest row of data using the saved model.
    """
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run training first.")
        
    scaler = joblib.load(scaler_path)
    
    # Load DL scaler if needed
    dl_scaler = None
    if model_type in ["lstm", "gru", "cnn1d"]:
        dl_scaler_path = os.path.join(models_dir, "dl_scaler.joblib")
        if os.path.exists(dl_scaler_path):
            dl_scaler = joblib.load(dl_scaler_path)
            
    # Preprocess
    X_latest, preprocessed_df = preprocess_new_data(history_df, scaler, dl_scaler=dl_scaler, window_size=window_size)
    
    # Load model
    if model_type in ["lstm", "gru", "cnn1d"]:
        # Deep Learning models require a sliding window sequence
        model_path = os.path.join(models_dir, f"{model_type}_best.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DL model not found at {model_path}")
            
        # Deep learning columns
        dl_cols = ['type_encoded', 'air_temperature_scaled', 'process_temperature_scaled',
                   'rotational_speed_scaled', 'torque_scaled', 'tool_wear_scaled']
        
        # Extract the last N steps (seq_len)
        seq_len = 10
        if len(preprocessed_df) < seq_len:
            # Pad with first row if history is not long enough
            pad_len = seq_len - len(preprocessed_df)
            first_row = preprocessed_df.iloc[[0] * pad_len]
            preprocessed_df = pd.concat([first_row, preprocessed_df], ignore_index=True)
            
        X_seq = preprocessed_df[dl_cols].values[-seq_len:].reshape(1, seq_len, len(dl_cols))
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        
        # Instantiate and load PyTorch model weights
        input_dim = len(dl_cols)
        if model_type == "lstm":
            model = LSTMClassifier(input_dim)
        elif model_type == "gru":
            model = GRUClassifier(input_dim)
        else:
            model = CNN1DClassifier(input_dim, seq_len=seq_len)
            
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        with torch.no_grad():
            logits = model(X_tensor)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob >= 0.5 else 0
            
    else:
        # Traditional ML models
        model_path = os.path.join(models_dir, f"{model_type}_best.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Traditional ML model not found at {model_path}")
            
        model = joblib.load(model_path)
        pred = int(model.predict(X_latest)[0])
        
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_latest)[0][1])
        else:
            prob = 1.0 if pred == 1 else 0.0
            
    return pred, prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict machine failure using time-series sensor data.")
    parser.add_argument("--model", type=str, default="xgboost", help="Model type to load (logistic_regression, random_forest, xgboost, lightgbm, lstm, gru, cnn1d, etc.)")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory where models are saved")
    args = parser.parse_args()
    
    # Create sample dummy history data (minimum 10 rows for rolling statistics and sequence length)
    logger.info("Generating dummy sensor readings for demonstration...")
    dummy_history = pd.DataFrame({
        'type': ['L'] * 10,
        'air_temperature': [298.1 + i*0.1 for i in range(10)],
        'process_temperature': [308.2 + i*0.1 for i in range(10)],
        'rotational_speed': [1500 - i*20 for i in range(10)],
        'torque': [40.0 + i*1.5 for i in range(10)],
        'tool_wear': [60 + i*2 for i in range(10)]
    })
    
    try:
        pred, prob = predict_single_step(dummy_history, models_dir=args.models_dir, model_type=args.model)
        logger.info(f"Prediction for latest sensor reading using {args.model}:")
        logger.info(f"  Failure Predicted: {'YES' if pred == 1 else 'NO'} (class {pred})")
        logger.info(f"  Probability of Failure: {prob:.4%}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}. Check if you have trained the model using train.py first.")
