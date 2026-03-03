import argparse
import joblib
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, confusion_matrix, 
    roc_curve, f1_score
)
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def load_model_bundle(path):
    print(f"[INFO] Loading model from {path}...")
    try:
        data = joblib.load(path)
        print(f"[INFO] Model loaded. Serial: {data.get('serial_number', 'unknown')}")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        exit(1)

def prepare_encoded_data(X, cat_cols):
    """Replicates encoding logic for Non-CatBoost models (XGB, LGBM, RF)."""
    X_enc = X.copy()
    for col in cat_cols:
        if col in X_enc.columns:
            # Check safely for categorical dtype
            if isinstance(X_enc[col].dtype, pd.CategoricalDtype):
                X_enc[col] = X_enc[col].cat.codes
            else:
                X_enc[col] = X_enc[col].astype('category').cat.codes
    return X_enc.fillna(-999)

def find_optimal_threshold_youden(y_true, y_probs):
    """Finds the threshold that maximizes Sensitivity + Specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]

def calculate_calibration_slope(y_true, y_probs):
    """Calculates Calibration Slope and Intercept using Logistic Regression."""
    eps = 1e-15
    y_probs_clipped = np.clip(y_probs, eps, 1 - eps)
    log_odds = np.log(y_probs_clipped / (1 - y_probs_clipped))
    
    lr = LogisticRegression(C=1e9, solver='lbfgs') 
    lr.fit(log_odds.reshape(-1, 1), y_true)
    
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

def calculate_net_benefit(y_true, y_probs, threshold):
    """Calculates Net Benefit at a specific threshold."""
    n = len(y_true)
    if threshold >= 1.0: threshold = 0.999
    if threshold <= 0.0: threshold = 0.001
    
    preds = (y_probs >= threshold).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    
    nb = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return nb

def compute_full_metrics(y_true, y_probs, model_name):
    """Generates all requested metrics for a single model."""
    
    # 1. Probabilistic Metrics
    auc = roc_auc_score(y_true, y_probs)
    brier = brier_score_loss(y_true, y_probs)
    try:
        cal_slope, cal_intercept = calculate_calibration_slope(y_true, y_probs)
    except:
        cal_slope, cal_intercept = 1.0, 0.0 # Fallback if singular
    
    # 2. Optimal Threshold (Youden's J)
    opt_thresh = find_optimal_threshold_youden(y_true, y_probs)
    
    # 3. Binary Metrics (at Optimal Threshold)
    preds = (y_probs >= opt_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, preds, zero_division=0)
    
    # 4. Clinical Utility
    nb = calculate_net_benefit(y_true, y_probs, opt_thresh)
    
    return {
        "Model": model_name,
        "AUROC": float(auc),
        "Calibration_Slope": float(cal_slope),
        "Calibration_Intercept": float(cal_intercept),
        "Brier_Score": float(brier),
        "Net_Benefit": float(nb),
        "Optimal_Threshold": float(opt_thresh),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "PPV": float(ppv),
        "NPV": float(npv),
        "F1_Score": float(f1)
    }

# ==========================================
# 2. MAIN LOGIC
# ==========================================

def main(model_path):
    # --- Load ---
    bundle = load_model_bundle(model_path)
    
    # Prepare Output
    serial = bundle.get('serial_number', 'unknown')
    output_dir = os.path.join("external_assessor", serial)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract Data
    X_test_raw = bundle.get('X_test_exact')
    y_test = bundle.get('y_test_exact')
    
    # Safety: Ensure y_test is numpy array
    if isinstance(y_test, (pd.Series, pd.DataFrame)):
        y_test = y_test.to_numpy().ravel()
        
    fold_models = bundle.get('fold_models')
    n_folds = bundle.get('n_folds', 5)
    cat_cols = bundle.get('cat_cols', [])
    feature_names = bundle.get('feature_names', [])

    # Ensure X_test_raw has column names if missing
    if not isinstance(X_test_raw, pd.DataFrame):
         X_test_raw = pd.DataFrame(X_test_raw, columns=feature_names)
    
    print(f"[INFO] Assessing {len(y_test)} patients across 4 ensemble architectures...")

    # --- Prepare Encoded Data (for XGB, LGBM, RF) ---
    X_test_enc = prepare_encoded_data(X_test_raw, cat_cols)
    
    results = {}
    
    # --- Loop Through Base Models ---
    for algo in ['cat', 'xgb', 'lgbm', 'rf']:
        print(f"   > Processing {algo.upper()}...")
        
        # Aggregate predictions across folds
        algo_preds = np.zeros(len(y_test))
        models = fold_models[algo]
        
        for model in models:
            if algo == 'cat':
                # CatBoost uses Raw Data (handles categories internally)
                preds = model.predict_proba(X_test_raw)[:, 1]
            else:
                # Others use Encoded Data
                # Handle XGBoost feature mismatch issues by using numpy array if names fail
                try:
                    preds = model.predict_proba(X_test_enc)[:, 1]
                except:
                    preds = model.predict_proba(X_test_enc.values)[:, 1]
                    
            algo_preds += preds
            
        # Average the predictions
        avg_preds = algo_preds / n_folds
        
        # Calculate Metrics
        metrics = compute_full_metrics(y_test, avg_preds, algo.upper())
        results[algo] = metrics

    # --- Export to JSON ---
    output_path = os.path.join(output_dir, "individual_model_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
        
    print(f"\n[SUCCESS] Metrics saved to: {output_path}")
    
    # --- Print Summary Table ---
    print("\n" + "="*110)
    print(f"{'Metric':<20} | {'CatBoost':<12} | {'XGBoost':<12} | {'LightGBM':<12} | {'RandomForest':<12}")
    print("-" * 110)
    
    keys = ["AUROC", "Calibration_Slope", "Brier_Score", "Net_Benefit", "Sensitivity", "Specificity", "PPV", "NPV", "F1_Score"]
    
    for k in keys:
        row = f"{k:<20} | "
        for algo in ['cat', 'xgb', 'lgbm', 'rf']:
            val = results[algo][k]
            row += f"{val:<12.4f} | "
        print(row)
    print("="*110)

if __name__ == "__main__":
    path = './outputs/03022026_115404_58666/model/03022026_115404_58666.pkl'
    main(path)