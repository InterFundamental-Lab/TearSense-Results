import joblib
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, confusion_matrix, 
    roc_curve, f1_score, RocCurveDisplay
)
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ==========================================
# JSON ENCODER FOR NUMPY TYPES
# ==========================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def load_model_bundle(path):
    print(f"[INFO] Loading model from {path}...")
    try:
        data = joblib.load(path)
        print(f"[INFO] Model loaded successfully. Version: {data.get('export_version', 'unknown')}")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        exit(1)


def prepare_encoded_data(X, cat_cols):
    """Replicates encoding logic for Tree Models (XGB/LGBM/RF)."""
    X_enc = X.copy()
    for col in cat_cols:
        if col in X_enc.columns:
            if isinstance(X_enc[col].dtype, pd.CategoricalDtype):
                X_enc[col] = X_enc[col].cat.codes
            else:
                X_enc[col] = X_enc[col].astype('category').cat.codes
    return X_enc.fillna(-999)


def generate_meta_features(base_preds, use_interactions=True, use_rank=False):
    """Reconstructs meta-features. MUST match defecator.meta_features exactly."""
    models = ['cat', 'xgb', 'lgbm', 'rf']
    
    clean = {}
    for m in models:
        c = np.nan_to_num(base_preds[m], nan=0.5)
        clean[m] = np.clip(c, 1e-7, 1 - 1e-7)
    
    X_meta = np.column_stack([clean[m] for m in models])
    
    if use_interactions:
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i < j:
                    X_meta = np.column_stack([X_meta, clean[m1] * clean[m2]])
        
        all_preds = np.array([clean[m] for m in models])
        X_meta = np.column_stack([
            X_meta,
            np.mean(all_preds, axis=0),
            np.std(all_preds, axis=0),
            np.max(all_preds, axis=0) - np.min(all_preds, axis=0),
        ])
    
    return X_meta


def calculate_net_benefit_curve(y_true, y_probs, thresholds):
    """Calculates Net Benefit curve for DCA plotting."""
    net_benefits = []
    n = len(y_true)
    for thresh in thresholds:
        if thresh >= 1.0: thresh = 0.999
        preds = (y_probs >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefits.append(nb)
    return net_benefits


def calculate_single_net_benefit(y_true, y_probs, threshold):
    """Calculates a scalar Net Benefit at a specific threshold."""
    n = len(y_true)
    if threshold >= 1.0: threshold = 0.999 
    preds = (y_probs >= threshold).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))


def find_optimal_threshold_youden(y_true, y_probs):
    """Find threshold that maximizes Youden's J."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def calculate_calibration_slope(y_true, y_probs):
    """Calculates Calibration Slope and Intercept via logistic regression on log-odds."""
    eps = 1e-15
    y_probs_clipped = np.clip(y_probs, eps, 1 - eps)
    log_odds = np.log(y_probs_clipped / (1 - y_probs_clipped))
    
    lr = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000)
    lr.fit(log_odds.reshape(-1, 1), y_true)
    
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def bootstrap_metrics(y_true, y_probs, threshold, n_bootstraps=1000, seed=42):
    """Bootstrap 95% Confidence Intervals for all metrics."""
    rng = np.random.RandomState(seed)
    
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    indices = np.arange(len(y_true))
    
    boot_stats = {k: [] for k in [
        "AUC", "Brier_Score", "F1_Score", "Sensitivity_Recall", 
        "Specificity", "PPV_Precision", "NPV", "Net_Benefit", "Calibration_Slope"
    ]}
    
    for _ in range(n_bootstraps):
        idx = rng.choice(indices, size=len(indices), replace=True)
        y_true_boot = y_true[idx]
        y_probs_boot = y_probs[idx]
        
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        preds_boot = (y_probs_boot >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_boot, preds_boot).ravel()
        
        boot_stats["AUC"].append(roc_auc_score(y_true_boot, y_probs_boot))
        boot_stats["Brier_Score"].append(brier_score_loss(y_true_boot, y_probs_boot))
        boot_stats["F1_Score"].append(f1_score(y_true_boot, preds_boot, zero_division=0))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        boot_stats["Sensitivity_Recall"].append(sens)
        boot_stats["Specificity"].append(spec)
        boot_stats["PPV_Precision"].append(ppv)
        boot_stats["NPV"].append(npv)
        
        boot_stats["Net_Benefit"].append(
            calculate_single_net_benefit(y_true_boot, y_probs_boot, threshold)
        )
        
        try:
            slope, _ = calculate_calibration_slope(y_true_boot, y_probs_boot)
            boot_stats["Calibration_Slope"].append(slope)
        except:
            pass
        
    ci_results = {}
    for key, values in boot_stats.items():
        if len(values) > 0:
            ci_results[key] = (np.percentile(values, 2.5), np.percentile(values, 97.5))
        else:
            ci_results[key] = (0.0, 0.0)
            
    return ci_results


# ==========================================
# 2. LOGISTIC REGRESSION BASELINE
# ==========================================

def prepare_data_for_lr(X, cat_cols):
    """Prepare data for Logistic Regression with one-hot encoding."""
    X = X.copy()
    
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna('Missing').astype(str)
    
    if cat_cols:
        existing_cats = [c for c in cat_cols if c in X.columns]
        if existing_cats:
            X = pd.get_dummies(X, columns=existing_cats, drop_first=True, dummy_na=False)
    
    for col in X.columns:
        if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            X[col] = X[col].fillna(X[col].median())
    
    return X


def align_columns(X_train, X_test):
    """Ensure train and test have same columns."""
    all_cols = list(set(X_train.columns) | set(X_test.columns))
    
    for col in all_cols:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0
    
    sorted_cols = sorted(all_cols)
    return X_train[sorted_cols], X_test[sorted_cols]


def train_lr_baseline(X_train, y_train, X_test, cat_cols, feature_names):
    """Train logistic regression baseline and return predictions."""
    X_train_subset = X_train[feature_names].copy()
    X_test_subset = X_test[feature_names].copy()
    
    X_train_enc = prepare_data_for_lr(X_train_subset, cat_cols)
    X_test_enc = prepare_data_for_lr(X_test_subset, cat_cols)
    
    X_train_enc, X_test_enc = align_columns(X_train_enc, X_test_enc)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)
    
    lr_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    lr_model.fit(X_train_scaled, y_train)
    
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Generate coefficient dataframe
    coefs = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]
    
    coef_df = pd.DataFrame({
        'Feature': list(X_train_enc.columns),
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs),
        'Odds_Ratio': np.exp(coefs)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Generate formula string
    formula_parts = [f"{intercept:.4f}"]
    for _, row in coef_df.head(10).iterrows():
        sign = "+" if row['Coefficient'] >= 0 else "-"
        formula_parts.append(f"{sign} {abs(row['Coefficient']):.4f} × {row['Feature']}")
    
    formula_str = "z = " + " ".join(formula_parts[:3])
    if len(formula_parts) > 3:
        formula_str += "\n    " + " ".join(formula_parts[3:6])
    if len(formula_parts) > 6:
        formula_str += "\n    " + " ".join(formula_parts[6:])
    if len(coef_df) > 10:
        formula_str += f"\n    + ... ({len(coef_df) - 10} more terms)"
    formula_str += "\n\nP(retear) = 1 / (1 + exp(-z))"
    
    return lr_probs, lr_model, coef_df, formula_str


def compute_lr_metrics(y_true, probs, threshold):
    """Compute all metrics for LR."""
    auc = roc_auc_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    cal_slope, cal_intercept = calculate_calibration_slope(y_true, probs)
    net_benefit = calculate_single_net_benefit(y_true, probs, threshold)
    
    return {
        "AUC": float(auc),
        "Brier_Score": float(brier),
        "F1_Score": float(f1_score(y_true, preds, zero_division=0)),
        "Sensitivity_Recall": float(tp / (tp + fn)) if (tp+fn) > 0 else 0.0,
        "Specificity": float(tn / (tn + fp)) if (tn+fp) > 0 else 0.0,
        "PPV_Precision": float(tp / (tp + fp)) if (tp+fp) > 0 else 0.0,
        "NPV": float(tn / (tn + fn)) if (tn+fn) > 0 else 0.0,
        "Net_Benefit": float(net_benefit),
        "Calibration_Slope": float(cal_slope),
        "Calibration_Intercept": float(cal_intercept),
        "Threshold_Used": float(threshold),
        "Confusion_Matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
    }


# ==========================================
# 3. MAIN LOGIC
# ==========================================

def main(model_path):
    bundle = load_model_bundle(model_path)
    
    serial_number = bundle.get('serial_number', 'unknown_serial')
    
    # Directories
    output_dir = os.path.join("external_assessor", serial_number)
    output_dir_auc = os.path.join("external_assessor", "auroc")
    output_dir_calibration = os.path.join("external_assessor", "calibration")
    output_dir_dca = os.path.join("external_assessor", "dca")
    output_dir_metrics = os.path.join("external_assessor", "metrics")

    for d in [output_dir, output_dir_auc, output_dir_calibration, output_dir_dca, output_dir_metrics]:
        os.makedirs(d, exist_ok=True)
    
    print(f"[INFO] Output Directory: {output_dir}")

    # Extract data
    X_test_raw = bundle.get('X_test_exact')
    y_test = bundle.get('y_test_exact')
    fold_models = bundle.get('fold_models')
    n_folds = bundle.get('n_folds', 5)
    cat_cols = bundle.get('cat_cols', [])
    feature_names = bundle.get('feature_names', [])
    
    # Training data for LR
    X_train = bundle.get('X_train_all')
    y_train = bundle.get('y_train_all')
    has_train_data = X_train is not None and y_train is not None
    
    stacking_config = bundle.get('stacking_config', {})
    use_interactions = stacking_config.get('use_interactions', True)
    use_rank = stacking_config.get('use_rank_features', False)
    
    stored_metrics = bundle.get('metrics', {})
    stored_auc = stored_metrics.get('test_auc')
    stored_brier = stored_metrics.get('test_brier')
    stored_threshold = bundle.get('optimal_threshold', 0.5)
    
    if X_test_raw is None or fold_models is None:
        print("[ERROR] Missing test data or models in .pkl")
        exit(1)

    if isinstance(y_test, (pd.Series, pd.DataFrame)):
        y_test = y_test.to_numpy().ravel()
    
    if has_train_data:
        y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        print(f"[INFO] Training data available: {len(y_train_arr)} samples for LR baseline")
    else:
        print("[WARNING] No training data in PKL. LR baseline will be skipped.")
    
    print(f"[INFO] Stacking config: use_interactions={use_interactions}, use_rank={use_rank}")
    print(f"[INFO] Stored threshold: {stored_threshold:.4f}")
    
    # ─────────────────────────────────────────
    # RUN TEARSENSE INFERENCE
    # ─────────────────────────────────────────
    print(f"\n[INFO] Running TearSense inference on {len(y_test)} patients...")
    X_test_enc = prepare_encoded_data(X_test_raw, cat_cols)
    base_preds = {}

    for algo in ['cat', 'xgb', 'lgbm', 'rf']:
        algo_preds = np.zeros(len(y_test))
        for model in fold_models[algo]:
            if algo == 'cat':
                preds = model.predict_proba(X_test_raw)[:, 1]
            else:
                preds = model.predict_proba(X_test_enc)[:, 1]
            algo_preds += preds
        base_preds[algo] = algo_preds / n_folds

    is_weighted_avg = bundle.get('is_weighted_avg', False)
    
    if is_weighted_avg:
        w = bundle.get('weights')
        if isinstance(w, dict):
            raw_final_probs = (w['cat']*base_preds['cat'] + w['xgb']*base_preds['xgb'] + 
                               w['lgbm']*base_preds['lgbm'] + w['rf']*base_preds['rf'])
        else:
            w = bundle.get('weights_array')
            raw_final_probs = sum(w[i] * base_preds[m] for i, m in enumerate(['cat', 'xgb', 'lgbm', 'rf']))
    else:
        X_meta = generate_meta_features(base_preds, use_interactions=use_interactions, use_rank=use_rank)
        meta_model = bundle.get('meta_model')
        raw_final_probs = meta_model.predict_proba(X_meta)[:, 1]

    calibrator = bundle.get('calibrator')
    ts_probs = calibrator.predict(raw_final_probs) if calibrator else raw_final_probs

    # Verify
    computed_auc = roc_auc_score(y_test, ts_probs)
    computed_brier = brier_score_loss(y_test, ts_probs)
    
    print(f"\n[CROSS-CHECK] Verifying TearSense inference...")
    print(f"  Computed AUC:   {computed_auc:.6f}  (stored: {stored_auc})")
    print(f"  Computed Brier: {computed_brier:.6f}  (stored: {stored_brier})")
    
    auc_match = abs(computed_auc - stored_auc) < 1e-4 if stored_auc else False
    brier_match = abs(computed_brier - stored_brier) < 1e-4 if stored_brier else False
    
    if auc_match and brier_match:
        print(f"  [PASS] Inference reproduces training evaluation exactly.")
    else:
        print(f"  [WARNING] Metrics don't match stored values!")

    # ─────────────────────────────────────────
    # TRAIN LR BASELINE
    # ─────────────────────────────────────────
    lr_probs = None
    lr_metrics = None
    lr_formula = None
    lr_coef_df = None
    
    if has_train_data:
        print(f"\n[INFO] Training Logistic Regression baseline...")
        lr_probs, lr_model, lr_coef_df, lr_formula = train_lr_baseline(
            X_train, y_train_arr, X_test_raw, cat_cols, feature_names
        )
        
        lr_threshold = find_optimal_threshold_youden(y_test, lr_probs)
        lr_metrics = compute_lr_metrics(y_test, lr_probs, lr_threshold)
        
        # Bootstrap CIs for LR
        print("[INFO] Bootstrapping LR confidence intervals...")
        lr_cis = bootstrap_metrics(y_test, lr_probs, lr_threshold, n_bootstraps=1000)
        for k, v in lr_cis.items():
            lr_metrics[f"{k}_95CI"] = [float(v[0]), float(v[1])]
        
        # Save coefficients
        coef_path = os.path.join(output_dir, f"lr_coefficients_{serial_number}.csv")
        lr_coef_df.to_csv(coef_path, index=False)
        print(f"[EXPORT] LR coefficients saved to: {coef_path}")

    # ─────────────────────────────────────────
    # COMPUTE TEARSENSE METRICS
    # ─────────────────────────────────────────
    youden_thresh = find_optimal_threshold_youden(y_test, ts_probs)
    
    preds_binary = (ts_probs >= youden_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_binary).ravel()

    cal_slope, cal_intercept = calculate_calibration_slope(y_test, ts_probs)
    net_benefit = calculate_single_net_benefit(y_test, ts_probs, youden_thresh)
    
    ts_metrics = {
        "model_serial": serial_number,
        "holdout_n": int(len(y_test)),
        "inference_verified": auc_match and brier_match,
        "stored_AUC": float(stored_auc) if stored_auc else None,
        "stored_Brier": float(stored_brier) if stored_brier else None,
        "stored_threshold": float(stored_threshold),
        "AUC": float(computed_auc),
        "Brier_Score": float(computed_brier),
        "Calibration_Slope": float(cal_slope),
        "Calibration_Intercept": float(cal_intercept),
        "Net_Benefit": float(net_benefit),
        "Optimal_Threshold_Used": float(youden_thresh),
        "Youden_J_Threshold": float(youden_thresh),
        "F1_Score": float(f1_score(y_test, preds_binary, zero_division=0)),
        "Sensitivity_Recall": float(tp / (tp + fn)) if (tp+fn) > 0 else 0.0,
        "Specificity": float(tn / (tn + fp)) if (tn+fp) > 0 else 0.0,
        "PPV_Precision": float(tp / (tp + fp)) if (tp+fp) > 0 else 0.0,
        "NPV": float(tn / (tn + fn)) if (tn+fn) > 0 else 0.0,
        "Confusion_Matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
    }

    # Bootstrap CIs
    print("[INFO] Bootstrapping TearSense confidence intervals...")
    ts_cis = bootstrap_metrics(y_test, ts_probs, youden_thresh, n_bootstraps=1000)
    
    for k, v in ts_cis.items():
        ts_metrics[f"{k}_95CI"] = [float(v[0]), float(v[1])]
        point_est = ts_metrics.get(k, 0.0)
        ts_metrics[f"{k}_Text"] = f"{point_est:.4f} ({v[0]:.4f}-{v[1]:.4f})"

    # Combined metrics output
    combined_metrics = {
        "tearsense": ts_metrics,
        "logistic_regression": lr_metrics if lr_metrics else "Not available (no training data)",
        "lr_formula": lr_formula if lr_formula else None,
    }
    
    if lr_metrics:
        combined_metrics["comparison"] = {
            "AUC_difference": float(ts_metrics['AUC'] - lr_metrics['AUC']),
            "Brier_difference": float(ts_metrics['Brier_Score'] - lr_metrics['Brier_Score']),
            "TearSense_better_AUC": ts_metrics['AUC'] > lr_metrics['AUC'],
            "TearSense_better_Brier": ts_metrics['Brier_Score'] < lr_metrics['Brier_Score'],
        }

    # Save JSON
    json_path = os.path.join(output_dir, f"clinical_metrics_{serial_number}.json")
    with open(json_path, 'w') as f:
        json.dump(combined_metrics, f, indent=4, cls=NumpyEncoder)
    print(f"[EXPORT] Metrics saved to {json_path}")

    json_path_combined = os.path.join(output_dir_metrics, f"{serial_number}.json")
    with open(json_path_combined, 'w') as f:
        json.dump(combined_metrics, f, indent=4, cls=NumpyEncoder)

    # ─────────────────────────────────────────
    # PRINT REPORT
    # ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CLINICAL METRICS REPORT — {serial_number}")
    print(f"{'='*70}")
    print(f"Inference Verified: {'✓ PASS' if ts_metrics['inference_verified'] else '✗ FAIL'}")
    print(f"Holdout N: {ts_metrics['holdout_n']}")
    print(f"TearSense Threshold (Youden): {youden_thresh:.4f}")
    
    if lr_metrics:
        print(f"LR Threshold (Youden): {lr_metrics['Threshold_Used']:.4f}")
    
    print(f"\n{'-'*70}")
    print(f"{'Metric':<25} | {'TearSense':>15} | {'LR Baseline':>15} | {'Δ':>10}")
    print(f"{'-'*70}")
    
    report_keys = [
        ("AUC", "AUC"),
        ("Brier_Score", "Brier_Score"),
        ("Calibration_Slope", "Calibration_Slope"),
        ("Net_Benefit", "Net_Benefit"),
        ("Sensitivity_Recall", "Sensitivity_Recall"),
        ("Specificity", "Specificity"),
        ("PPV_Precision", "PPV_Precision"),
        ("NPV", "NPV"),
        ("F1_Score", "F1_Score")
    ]
    
    for ts_key, lr_key in report_keys:
        ts_val = ts_metrics[ts_key]
        if lr_metrics:
            lr_val = lr_metrics[lr_key]
            diff = ts_val - lr_val
            print(f"{ts_key:<25} | {ts_val:>15.4f} | {lr_val:>15.4f} | {diff:>+10.4f}")
        else:
            print(f"{ts_key:<25} | {ts_val:>15.4f} | {'N/A':>15} | {'N/A':>10}")
    
    print(f"{'-'*70}")
    
    if lr_metrics:
        print(f"\n>> TearSense vs LR Baseline:")
        if ts_metrics['AUC'] > lr_metrics['AUC']:
            improvement = (ts_metrics['AUC'] - lr_metrics['AUC']) / lr_metrics['AUC'] * 100
            print(f"   ✓ TearSense AUC is {improvement:.2f}% better than LR")
        else:
            print(f"   ✗ LR AUC is better than TearSense")
    
    print(f"{'='*70}")

    # ─────────────────────────────────────────
    # COMPARISON PLOTS
    # ─────────────────────────────────────────
    print("\n[INFO] Generating comparison plots...")
    
    # 1. ROC COMPARISON
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # TearSense ROC
    RocCurveDisplay.from_predictions(
        y_test, ts_probs, 
        name=f"TearSense (AUC={ts_metrics['AUC']:.3f})",
        color="darkorange", ax=ax
    )
    
    # LR ROC
    if lr_probs is not None:
        RocCurveDisplay.from_predictions(
            y_test, lr_probs,
            name=f"Logistic Regression (AUC={lr_metrics['AUC']:.3f})",
            color="blue", ax=ax
        )
    
    ax.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.5)")
    ax.set_title(f"ROC Comparison — {serial_number}")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="lower right")
    
    plt.savefig(os.path.join(output_dir, f"auc_comparison_{serial_number}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir_auc, f"auc_comparison_{serial_number}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. CALIBRATION COMPARISON
    fig, ax = plt.subplots(figsize=(8, 8))
    
    CalibrationDisplay.from_predictions(y_test, ts_probs, n_bins=10, name="TearSense", ax=ax)
    if lr_probs is not None:
        CalibrationDisplay.from_predictions(y_test, lr_probs, n_bins=10, name="Logistic Regression", ax=ax)
    
    ax.set_title(f"Calibration Comparison — {serial_number}")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(output_dir, f"calibration_comparison_{serial_number}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir_calibration, f"calibration_comparison_{serial_number}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. DCA COMPARISON
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds_dca = np.linspace(0.01, 0.99, 100)
    
    # TearSense DCA
    nb_ts = calculate_net_benefit_curve(y_test, ts_probs, thresholds_dca)
    ax.plot(thresholds_dca, nb_ts, color='darkorange', lw=2, label='TearSense')
    
    # LR DCA
    if lr_probs is not None:
        nb_lr = calculate_net_benefit_curve(y_test, lr_probs, thresholds_dca)
        ax.plot(thresholds_dca, nb_lr, color='blue', lw=2, label='Logistic Regression')
    
    # Reference lines
    prevalence = np.mean(y_test)
    nb_all = prevalence - (1 - prevalence) * (thresholds_dca / (1 - thresholds_dca))
    nb_none = np.zeros_like(thresholds_dca)
    
    ax.plot(thresholds_dca, nb_all, color='gray', linestyle='--', label='Treat All')
    ax.plot(thresholds_dca, nb_none, color='black', linestyle=':', label='Treat None')
    
    ax.set_ylim(-0.05, max(prevalence, max(nb_ts)) + 0.05)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(f"Decision Curve Analysis — {serial_number}")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(output_dir, f"dca_comparison_{serial_number}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir_dca, f"dca_comparison_{serial_number}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[EXPORT] Plots saved to {output_dir}/")
    print("[DONE]")
    
    return combined_metrics, ts_probs, lr_probs


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="External assessment with LR baseline")
    # parser.add_argument('--model_path', type=str, required=True, help="Path to .pkl file")
    # args = parser.parse_args()
    # main(args.model_path)

    serials_to_run = [
        '03022026_115404_58666',
        '05022026_123748_52767',
        '05022026_124840_71444',
        '05022026_125050_98738',
        '05022026_125524_66946',
        '05022026_130008_98590',
        '05022026_130812_65089',
        '05022026_130839_45739',
        '05022026_131528_62930',
        '05022026_132823_67982',
        '05022026_134818_29477',
        '05022026_140401_57655',
    ]

    for serial in serials_to_run:
        model_path = f'outputs/{serial}/model/{serial}.pkl'
        main(os.path.join('./', model_path))