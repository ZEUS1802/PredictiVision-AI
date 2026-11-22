import numpy as np
import pandas as pd
import logging

def engineer_features_df(df, window_size=5):
    """
    Create engineered features for traditional ML models.
    Includes physical interaction features (temp diff, power, overstrain product),
    rolling statistics, and lag features.
    """
    # Create a copy to prevent SettingWithCopyWarning
    df_feat = df.copy()
    
    # 1. Physics-based features (derived from domain knowledge of AI4I dataset rules)
    # Heat Dissipation depends on difference between Process Temp and Air Temp
    df_feat['temp_difference'] = df_feat['process_temperature'] - df_feat['air_temperature']
    # Power = Torque * Rotational Speed (proportional to mechanical power)
    df_feat['power_index'] = df_feat['rotational_speed'] * df_feat['torque']
    # Overstrain depends on Tool Wear * Torque
    df_feat['overstrain_index'] = df_feat['tool_wear'] * df_feat['torque']
    
    # 2. Rolling statistics (mean and std) for key sensor readings
    cols_to_roll = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
    
    for col in cols_to_roll:
        df_feat[f'{col}_roll_mean_{window_size}'] = df_feat[col].rolling(window=window_size, min_periods=1).mean()
        df_feat[f'{col}_roll_std_{window_size}'] = df_feat[col].rolling(window=window_size, min_periods=1).std().fillna(0.0)
        
    # 3. Lag features (t-1 and t-2) for dynamic sensor readings
    cols_to_lag = ['rotational_speed', 'torque', 'air_temperature', 'temp_difference']
    
    for col in cols_to_lag:
        df_feat[f'{col}_lag_1'] = df_feat[col].shift(1).bfill()
        df_feat[f'{col}_lag_2'] = df_feat[col].shift(2).bfill()
        
    logging.info(f"Feature engineering completed. Added {df_feat.shape[1] - df.shape[1]} new features.")
    return df_feat

def create_sliding_windows(features, labels, window_size=10):
    """
    Generate sliding window sequences for Deep Learning (LSTM, GRU, 1D CNN).
    
    Parameters:
        features (np.ndarray): 2D array of shape (num_samples, num_features)
        labels (np.ndarray): 1D array of shape (num_samples,)
        window_size (int): length of the sequence window
        
    Returns:
        X (np.ndarray): 3D array of shape (num_sequences, window_size, num_features)
        y (np.ndarray): 1D array of shape (num_sequences,)
    """
    X, y = [], []
    for i in range(len(features) - window_size + 1):
        X.append(features[i : i + window_size])
        y.append(labels[i + window_size - 1])
    
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    
    return X_arr, y_arr

if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()
    
    # Simple test for feature engineering
    data = {
        'air_temperature': np.random.uniform(295, 305, 20),
        'process_temperature': np.random.uniform(305, 315, 20),
        'rotational_speed': np.random.uniform(1200, 2000, 20),
        'torque': np.random.uniform(10, 60, 20),
        'tool_wear': np.random.uniform(0, 100, 20)
    }
    df = pd.DataFrame(data)
    df_engineered = engineer_features_df(df, window_size=3)
    
    features_arr = df_engineered[['air_temperature', 'torque']].values
    labels_arr = np.random.randint(0, 2, 20)
    X, y = create_sliding_windows(features_arr, labels_arr, window_size=5)
    print("Engineered columns:", list(df_engineered.columns))
    print("Sliding windows shape - X:", X.shape, "y:", y.shape)
