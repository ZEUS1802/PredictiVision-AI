import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import logging

def evaluate_model_performance(y_true, y_pred, y_prob, model_name, figures_dir="figures"):
    """
    Evaluate model predictions and save Confusion Matrix, ROC, and PR curves.
    
    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted binary labels.
        y_prob (np.ndarray): Predicted probability of failure.
        model_name (str): Name of the model (for labeling plots).
        figures_dir (str): Directory to save plots.
        
    Returns:
        metrics (dict): Dictionary of evaluation metrics.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0
        
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }
    
    logging.info(
        f"--- {model_name} Evaluation ---\n"
        f"Accuracy:  {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1 Score:  {f1:.4f}\n"
        f"ROC AUC:   {roc_auc:.4f}\n"
        f"PR AUC:    {pr_auc:.4f}\n"
    )
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Failure'],
                yticklabels=['Normal', 'Failure'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(f"{model_name} - Confusion Matrix", fontsize=14, pad=15)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # 2. ROC & PR Curves Subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title(f'{model_name} - Receiver Operating Characteristic', fontsize=13, pad=10)
    axes[0].legend(loc="lower right", fontsize=11)
    
    # PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    # Average precision
    axes[1].plot(rec, prec, color='forestgreen', lw=2.5, label=f'PR Curve (AP/AUC = {pr_auc:.3f})')
    # Baseline is fraction of positives
    baseline = np.sum(y_true) / len(y_true)
    axes[1].axhline(y=baseline, color='red', lw=1.5, linestyle='--', label=f'Baseline ({baseline:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title(f'{model_name} - Precision-Recall Curve', fontsize=13, pad=10)
    axes[1].legend(loc="lower left", fontsize=11)
    
    plt.suptitle(f"{model_name} - Performance Curves", fontsize=16, y=0.98)
    plt.tight_layout()
    curves_path = os.path.join(figures_dir, f"{model_name.lower().replace(' ', '_')}_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    
    logging.info(f"Saved evaluation plots to {figures_dir}/")
    return metrics

def compare_models(metrics_list, figures_dir="figures"):
    """
    Compare multiple models using a bar plot and table.
    
    Parameters:
        metrics_list (list): List of dictionaries containing model metrics.
        figures_dir (str): Directory to save the comparison plot.
    """
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.set_index("model_name", inplace=True)
    
    # Save CSV of comparisons
    os.makedirs(figures_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(figures_dir, "model_comparison.csv"))
    
    # Plot comparisons
    plt.figure(figsize=(12, 7))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    
    df_melted = df_metrics[metrics_to_plot].reset_index().melt(
        id_vars="model_name", 
        var_name="Metric", 
        value_name="Score"
    )
    
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        data=df_melted, 
        x="Metric", 
        y="Score", 
        hue="model_name", 
        palette="viridis"
    )
    
    plt.title("Model Comparison - Key Performance Metrics", fontsize=16, pad=15)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Metric", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title="Models", loc="lower right", bbox_to_anchor=(1, 0.05), fontsize=10)
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=8)
            
    plt.tight_layout()
    comparison_plot_path = os.path.join(figures_dir, "model_comparison_bar.png")
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close()
    
    logging.info("Model comparison completed and saved comparison table & plot.")
    return df_metrics
