import os
import zipfile
import urllib.request
import logging
import random
import numpy as np
import torch

def setup_logging(log_file="predictive_maintenance.log"):
    """Set up logging configuration to print to console and write to log file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("PredictiveMaintenance")

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seeds set to {seed}")

def download_dataset(data_dir="data"):
    """Download the AI4I 2020 Predictive Maintenance Dataset from UCI Repository if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "ai4i2020.csv")
    
    if os.path.exists(csv_path):
        logging.info(f"Dataset already exists at {csv_path}")
        return csv_path
        
    url = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
    zip_path = os.path.join(data_dir, "dataset.zip")
    
    logging.info(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        logging.info("Download completed. Extracting zip file...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List files inside the zip
            file_names = zip_ref.namelist()
            logging.info(f"Files found in zip: {file_names}")
            
            # The CSV file inside the zip might be named 'ai4i2020.csv'
            for file_name in file_names:
                if file_name.endswith('.csv'):
                    zip_ref.extract(file_name, data_dir)
                    extracted_path = os.path.join(data_dir, file_name)
                    # If it has a different name, rename it to ai4i2020.csv
                    if file_name != "ai4i2020.csv":
                        os.rename(extracted_path, csv_path)
                    break
        
        # Clean up zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        logging.info(f"Dataset successfully unzipped and saved to {csv_path}")
        return csv_path
        
    except Exception as e:
        logging.error(f"Error downloading or extracting dataset: {e}")
        # Fallback to kaggle or other mirror link if needed, but UCI is primary
        raise e

if __name__ == "__main__":
    setup_logging()
    download_dataset()
