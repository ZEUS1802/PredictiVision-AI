import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import logging

# Set up logger
logger = logging.getLogger("PredictiveMaintenance")

def plot_tree_feature_importance(model, feature_names, model_name, save_path="figures/feature_importance.png"):
    """
    Plot and save traditional tree-based feature importance.
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"Model {model_name} does not have feature_importances_ attribute.")
        return None
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    df_imp = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis')
    plt.title(f"Feature Importance - {model_name}", fontsize=14, pad=15)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Tree-based feature importance plot saved to {save_path}")
    return df_imp

def plot_permutation_importance_score(model, X_val, y_val, feature_names, model_name, save_path="figures/permutation_importance.png"):
    """
    Calculate and plot permutation feature importance.
    """
    logger.info(f"Calculating permutation importance for {model_name}...")
    result = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)
    
    importances_mean = result.importances_mean
    indices = np.argsort(importances_mean)[::-1]
    
    df_perm = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance Mean': importances_mean[indices],
        'Importance Std': result.importances_std[indices]
    })
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    # Plot horizontal bar chart with error bars
    plt.barh(df_perm['Feature'][::-1], df_perm['Importance Mean'][::-1], 
             xerr=df_perm['Importance Std'][::-1], color='steelblue', align='center')
    plt.title(f"Permutation Feature Importance - {model_name}", fontsize=14, pad=15)
    plt.xlabel("Decrease in F1 Score / Metric Accuracy", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Permutation feature importance plot saved to {save_path}")
    return df_perm

def plot_shap_explanations(model, X_train, X_val, feature_names, save_dir="figures"):
    """
    Compute and save SHAP values summary plot and bar plot for tree-based models.
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP library is not installed. Skipping SHAP visualization.")
        return None
        
    logger.info("Computing SHAP values (using a sample from training/validation to speed up)...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Format features as DataFrame with correct names for SHAP
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    
    # Use TreeExplainer for tree models
    try:
        explainer = shap.TreeExplainer(model)
        # Compute shap values on validation set
        # Sample 200 samples from X_val to run quickly
        sample_size = min(200, len(X_val_df))
        X_sample = X_val_df.sample(n=sample_size, random_state=42)
        
        shap_values = explainer(X_sample)
        
        # 1. SHAP Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("SHAP Summary Plot", fontsize=14, pad=15)
        plt.tight_layout()
        summary_path = os.path.join(save_dir, "shap_summary_plot.png")
        plt.savefig(summary_path, dpi=300)
        plt.close()
        
        # 2. SHAP Feature Importance Bar Plot
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        plt.title("SHAP Feature Importance Bar Plot", fontsize=14, pad=15)
        plt.tight_layout()
        bar_path = os.path.join(save_dir, "shap_bar_plot.png")
        plt.savefig(bar_path, dpi=300)
        plt.close()
        
        logger.info(f"SHAP plots successfully saved to {save_dir}")
        return shap_values
        
    except Exception as e:
        logger.error(f"Failed to generate SHAP values: {e}. Fallback to generic Explainer.")
        try:
            # Fallback to kernel explainer or general explainer
            sample_size = min(100, len(X_val_df))
            X_sample = X_val_df.sample(n=sample_size, random_state=42)
            explainer = shap.Explainer(model, X_train_df.sample(n=50, random_state=42))
            shap_values = explainer(X_sample)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title("SHAP Summary Plot (General Explainer)", fontsize=14, pad=15)
            plt.tight_layout()
            summary_path = os.path.join(save_dir, "shap_summary_plot.png")
            plt.savefig(summary_path, dpi=300)
            plt.close()
            
            logger.info(f"SHAP summary plot successfully saved via generic Explainer.")
            return shap_values
        except Exception as ex:
            logger.error(f"SHAP general explainer also failed: {ex}")
            return None
