import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging

def load_raw_data(csv_path):
    """Load the raw dataset and clean column names."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    logging.info(f"Raw data loaded from {csv_path}. Shape: {df.shape}")
    
    # Rename columns to a clean lower-case format
    col_mapping = {
        'UDI': 'udi',
        'Product ID': 'product_id',
        'Type': 'type',
        'Air temperature [K]': 'air_temperature',
        'Process temperature [K]': 'process_temperature',
        'Rotational speed [rpm]': 'rotational_speed',
        'Torque [Nm]': 'torque',
        'Tool wear [min]': 'tool_wear',
        'Machine failure': 'machine_failure',
        'TWF': 'twf',
        'HDF': 'hdf',
        'PWF': 'pwf',
        'OSF': 'osf',
        'RNF': 'rnf'
    }
    df = df.rename(columns=col_mapping)
    return df

def clean_data(df):
    """Check for missing values, duplicates, and handle them."""
    # Check duplicates (excluding UDI)
    duplicate_count = df.duplicated(subset=['product_id', 'air_temperature', 'process_temperature', 
                                            'rotational_speed', 'torque', 'tool_wear']).sum()
    if duplicate_count > 0:
        logging.warning(f"Found {duplicate_count} duplicate rows. Removing duplicates.")
        df = df.drop_duplicates(subset=['product_id', 'air_temperature', 'process_temperature', 
                                        'rotational_speed', 'torque', 'tool_wear'])
    else:
        logging.info("No duplicates found.")
        
    # Check missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        logging.warning(f"Found {missing_values} missing values. Filling with median/mode.")
        # Fill numeric with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        # Fill categoricals with mode
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
    else:
        logging.info("No missing values found.")
        
    return df

def detect_outliers(df, cols, threshold=3):
    """Detect outliers using Z-score method for descriptive logging."""
    outlier_info = {}
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        z_scores = (df[col] - mean) / std
        outliers = np.abs(z_scores) > threshold
        outlier_info[col] = outliers.sum()
        logging.info(f"Outliers detected in '{col}' (Z > {threshold}): {outliers.sum()} instances")
    return outlier_info

def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data into train, validation, and test sets preserving chronological order."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9, "Split ratios must sum to 1"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logging.info(f"Chronological split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def preprocess_pipeline(train_df, val_df, test_df, models_dir="models"):
    """Fit encoders/scalers on train and transform all sets."""
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Categorical Encoding (Type)
    # Type maps to L/M/H. We can label encode or manually map L -> 0, M -> 1, H -> 2
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    
    for df in [train_df, val_df, test_df]:
        df['type_encoded'] = df['type'].map(type_mapping)
        
    # 2. Scaling numerical features
    num_features = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
    
    scaler = StandardScaler()
    
    # Fit on train numerical features
    train_scaled_features = scaler.fit_transform(train_df[num_features])
    val_scaled_features = scaler.transform(val_df[num_features])
    test_scaled_features = scaler.transform(test_df[num_features])
    
    # Save the scaler
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logging.info(f"Fitted scaler saved to {scaler_path}")
    
    # Add scaled features to DataFrames
    scaled_cols = [f"{col}_scaled" for col in num_features]
    
    train_df[scaled_cols] = train_scaled_features
    val_df[scaled_cols] = val_scaled_features
    test_df[scaled_cols] = test_scaled_features
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()
    
    # Quick execution test
    try:
        df = load_raw_data("data/ai4i2020.csv")
        df = clean_data(df)
        detect_outliers(df, ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear'])
        train_df, val_df, test_df = split_data(df)
        train_df, val_df, test_df = preprocess_pipeline(train_df, val_df, test_df)
        print("Preprocessing successfully completed.")
    except Exception as e:
        print(f"Test run aborted: {e}. Run utils.py first to download the dataset.")
