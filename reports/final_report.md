# Manufacturing Quality Control & Predictive Maintenance Final Report

## 1. Executive Summary & Problem Statement

In modern manufacturing, unexpected machinery failure is one of the primary drivers of unplanned downtime, leading to substantial production losses, increased repair costs, and safety hazards. Traditionally, factories operated under **run-to-failure (reactive)** or **scheduled (preventative)** maintenance regimes. However:
- **Reactive maintenance** triggers high recovery costs and secondary system damage.
- **Preventative maintenance** often results in premature replacements of components, wasting operational lifespan.

**Predictive Maintenance (PdM)** leveraging time-series sensor data and machine learning offers a middle path: it forecasts failures before they happen. This project builds and compares multiple Machine Learning and Deep Learning architectures utilizing the **AI4I 2020 Predictive Maintenance Dataset** to identify failures before they manifest, estimate machine health, and isolate the key sensor factors that signal impending faults.

---

## 2. Dataset Description

The dataset consists of **10,000 rows** of sensor records capturing continuous operation at 10-minute intervals:
- **UDI / Product ID**: Identifiers.
- **Type**: Quality variant of the tool (L: Low (50% of data), M: Medium (30%), H: High (20%)).
- **Air temperature [K]**: Ranging from 295.1 to 304.5 Kelvin.
- **Process temperature [K]**: Ranging from 305.7 to 313.8 Kelvin.
- **Rotational speed [rpm]**: Spindle rotation, exhibiting high variance (1168 to 2886 rpm).
- **Torque [Nm]**: Spindle torque (3.8 to 76.6 Nm).
- **Tool wear [min]**: Duration the tool has been used (0 to 246 minutes).
- **Machine failure**: Binary target label indicating whether the machine failed (3.39% frequency, representing a highly imbalanced classification task).
- **Failure Modes**: Independent labels indicating specific causes: Tool Wear (TWF), Heat Dissipation (HDF), Power (PWF), Overstrain (OSF), and Random Failures (RNF).

---

## 3. Data Cleaning & Preprocessing

1. **Duplicate Detection**: A subset check on sensor columns confirmed zero duplicate instances.
2. **Missing Value Handling**: A robust pipeline checked and confirmed no missing values, but stands ready with median/mode backfills.
3. **Outlier Analysis**: Evaluated using Z-score (threshold > 3):
   - **Rotational speed**: 189 outliers (extremely high rotational speed events).
   - **Torque**: 69 outliers (extreme load events).
   - *Decision*: Outliers were retained because in predictive maintenance, extreme sensor fluctuations (e.g., sudden speed or torque spikes) often serve as early indicators of anomalies and mechanical failure.
4. **Encoding**: Categorical variable `Type` (L, M, H) was label encoded to `type_encoded` (0, 1, 2).
5. **Feature Scaling**: Scaled using `StandardScaler` fitted exclusively on the training set to prevent temporal leakage into validation/testing sets.

---

## 4. Feature Engineering

To assist traditional ML models in capturing the physical relationships of the machine and its temporal state, we engineered:

### A. Physics-Based Features
- **Temperature Difference**: `temp_difference = process_temperature - air_temperature`. Used to identify Heat Dissipation Failure (HDF), which triggers when the process-to-air differential drops below 8.6 K under slow rotational speeds.
- **Power Index**: `power_index = rotational_speed * torque`. Captures mechanical power. Fits Power Failure (PWF) cases (occurring when power is under 3500W or over 9000W).
- **Overstrain Index**: `overstrain_index = tool_wear * torque`. Corresponds to Overstrain Failure (OSF), which occurs when the product of wear and torque exceeds structural limits.

### B. Time-Series / Lag Features
- **Rolling Averages**: Moving means and standard deviations (window size = 5) computed for temperatures, speed, torque, and wear to capture smoothing trends and volatility.
- **Lags (t-1, t-2)**: Lag variables for speed, torque, and temperatures to capture directional rate of change (e.g., deceleration or rapid heating).

### C. Sequential Sequence Generation
For Deep Learning models, raw scaled sensor records were transformed into 3D tensors of shape `(sequences, window_size, core_features)` with `window_size = 10` (representing a 100-minute historical sensor profile).

---

## 5. Machine Learning Models & Architecture

We compared two paradigms:

### A. Traditional Machine Learning
- **Logistic Regression**: Linear baseline with balanced class weights.
- **Decision Tree & Random Forest**: Tree models optimized using `TimeSeriesSplit` cross-validation.
- **XGBoost & LightGBM**: Gradient Boosted Trees. Handled severe class imbalance (3.39% positive) using `scale_pos_weight = 28.5`.
- **Support Vector Machine (SVM)**: Radial Basis Function (RBF) kernel with balanced class weights.

### B. Deep Learning (PyTorch)
These architectures automatically learn temporal features directly from sequence data:
- **LSTM**: Two-layer LSTM (64 hidden units) mapping to a fully-connected output.
- **GRU**: Two-layer GRU (64 hidden units) mapping to a fully-connected output.
- **1D CNN**: Two 1D convolution layers (32 & 64 filters) extracting spatial-temporal features, followed by global average pooling and a fully connected classifier.
- *Loss*: Trained using `BCEWithLogitsLoss` with a positive class weight modifier (`pos_weight = 28.5`) to combat minority imbalance.

---

## 6. Experimental Results & Performance Comparison

The models were evaluated on an independent chronologically partitioned test set (final 1,500 samples):

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Tuned)** | **99.33%** | **91.30%** | **72.41%** | **0.8077** | **91.66%** | **76.35%** |
| **LightGBM (Tuned)** | 99.20% | 84.00% | 72.41% | 0.7778 | 92.14% | 75.27% |
| **Random Forest (Tuned)**| 98.93% | 72.41% | 72.41% | 0.7241 | 95.76% | 73.64% |
| **Decision Tree** | 97.67% | 44.00% | 75.86% | 0.5570 | 81.60% | 57.80% |
| **1D CNN** | 98.12% | 50.00% | 57.14% | 0.5333 | 96.49% | 45.31% |
| **GRU** | 97.38% | 39.22% | 71.43% | 0.5063 | 97.36% | 53.99% |
| **SVM** | 96.73% | 33.33% | 68.97% | 0.4494 | 93.65% | 60.72% |
| **LSTM** | 94.17% | 21.36% | 78.57% | 0.3359 | 96.80% | 54.88% |
| **Logistic Regression** | 94.53% | 20.88% | 65.52% | 0.3167 | 90.45% | 36.41% |

### Key Observations:
1. **XGBoost** achieved the highest F1-Score (0.8077) and PR-AUC (0.7635). It maintained a high precision (91.30%) while catching 72.41% of failures.
2. **Deep Learning (LSTM & GRU)** demonstrated excellent ROC-AUC scores (>96.8%) but suffered from a higher rate of false positives (lower precision), leading to lower F1-scores. This is likely because the dataset is structured with strict deterministic boolean logic rules for failures, which tree-based algorithms can easily represent via recursive splits, whereas neural networks smooth out these decision boundaries.
3. **Recall Optimization**: The LSTM model achieved the highest raw Recall (78.57%), which is very desirable in predictive maintenance scenarios.

---

## 7. The Crucial Importance of Recall in Predictive Maintenance

In standard classification, Accuracy and F1-Score are primary goals. However, in industrial Predictive Maintenance, **Recall is the critical metric**.

Recall is defined as:
$$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$

- **True Positive (TP)**: Model predicts failure, and machine is indeed failing. Maintenance is scheduled.
- **False Negative (FN)**: Model predicts normal, but machine fails. This is a **missed failure**.
- **False Positive (FP)**: Model predicts failure, but machine is normal. This is a **false alarm**.

### Cost Asymmetry in PdM
In manufacturing:
- **Cost of a False Negative (FN)**: Extremely high. It results in unexpected machine breakdowns, damaged equipment, halted assembly lines, lost output, and potentially safety incidents. The cost can range from tens of thousands to millions of dollars.
- **Cost of a False Positive (FP)**: Low. It leads to a technician performing a brief inspection on a functioning machine. The cost is limited to an hour of labor and minor inspection overhead.

Because the cost of a missed failure ($FN$) is orders of magnitude larger than the cost of a false alarm ($FP$), models that maximize **Recall** (even at the cost of some Precision) are far more valuable to businesses. The LSTM model (Recall: 78.57%) and Tuned XGBoost (Recall: 72.41%, Precision: 91.30%) represent the two best candidates for deployment depending on the operator's tolerance for false alarms.

---

## 8. Explainable AI (XAI) Analysis

Using **XGBoost** (the best performing model) as our interpreter:

1. **Tree-Based Feature Importance**:
   - The physics-based engineered feature `overstrain_index` (Tool Wear * Torque) and `power_index` (Speed * Torque) emerged as the single most critical predictors.
   - This aligns with the physical boundaries of OSF (Overstrain Failure) and PWF (Power Failure) within the AI4I dataset rules.

2. **Permutation Importance**:
   - Shuffling the values of `overstrain_index` and `torque` resulted in the largest drop in test metric performance.
   - This validates that the model relies heavily on mechanical torque load indicators to classify failure states.

3. **SHAP (SHapley Additive exPlanations) Values**:
   - **Summary Plot**: High values of `overstrain_index` (red points pushed to the right) have a massive positive SHAP value, indicating they strongly push the model output toward predicting failure.
   - Conversely, very high values of `temp_difference` (process minus air temperature) reduce the likelihood of failure, whereas low differentials (less than 8.6 K) act as warning signals, raising the probability of Heat Dissipation Failure (HDF).

---

## 9. Limitations & Future Work

### Limitations:
- **Synthetic Data**: The AI4I 2020 dataset is synthetic. Real-world machinery sensor streams exhibit more complex noise, sensor drift, and multi-variable dependencies.
- **Fixed Window Size**: Sequential sequences are limited to a fixed 10-step history. In real plants, failures can develop over days or weeks.

### Future Work:
1. **Semi-Supervised Anomaly Detection**: Utilizing Autoencoders or Isolation Forests to identify unknown failure modes where labeled failure samples are unavailable.
2. **Remaining Useful Life (RUL) Prediction**: Transitioning from binary failure classification to continuous regression modeling to predict exactly how many hours of operation remain for a tool.
3. **Multi-Task Learning**: Building a model that predicts both machine failure (binary classification) and the specific failure type (multi-label classification) simultaneously.
