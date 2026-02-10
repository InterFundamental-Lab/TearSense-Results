import argparse
import joblib
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, confusion_matrix, 
    f1_score, roc_curve, RocCurveDisplay
)
from sklearn.calibration import CalibrationDisplay


# ==========================================
# JSON ENCODER
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


# ==========================================
# METRIC FUNCTIONS
# ==========================================

def find_youden_threshold(y_true, y_probs):
    """Find threshold maximizing Youden's J."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def calculate_net_benefit(y_true, y_prob, threshold):
    """Calculate net benefit at threshold."""
    if threshold <= 0 or threshold >= 1:
        return 0.0
    tp = np.sum((y_prob >= threshold) & (y_true == 1))
    fp = np.sum((y_prob >= threshold) & (y_true == 0))
    n = len(y_true)
    weight = threshold / (1 - threshold)
    return (tp / n) - (fp / n) * weight


def calculate_net_benefit_curve(y_true, y_probs, thresholds):
    """Calculate net benefit curve for DCA."""
    return [calculate_net_benefit(y_true, y_probs, t) for t in thresholds]


def get_calibration_slope_intercept(y_true, y_prob):
    """Calculate calibration slope and intercept."""
    from sklearn.linear_model import LogisticRegression as LR
    epsilon = 1e-10
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    log_odds = np.log(y_prob / (1 - y_prob))
    
    clf = LR(C=1e9, solver='lbfgs', max_iter=1000)
    clf.fit(log_odds.reshape(-1, 1), y_true)
    
    return float(clf.coef_[0][0]), float(clf.intercept_[0])


def compute_all_metrics(y_true, probs, threshold):
    """Compute all clinical metrics."""
    auc = roc_auc_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, preds, zero_division=0)
    
    net_benefit = calculate_net_benefit(y_true, probs, threshold)
    cal_slope, cal_intercept = get_calibration_slope_intercept(y_true, probs)
    
    return {
        "AUC": float(auc),
        "Brier": float(brier),
        "F1_Score": float(f1),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "PPV": float(ppv),
        "NPV": float(npv),
        "Net_Benefit": float(net_benefit),
        "Calibration_Slope": float(cal_slope),
        "Calibration_Intercept": float(cal_intercept),
        "Threshold_Used": float(threshold),
        "Confusion_Matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
    }


# ==========================================
# DATA PREPARATION
# ==========================================

def reconstruct_data_from_csv(feature_names, cat_cols, test_size=0.2, feature_engineering=False):
    """
    Reconstruct training data from training.csv using the same logic as core.py.
    Uses random_state=42 to get the exact same train/test split.
    """
    print("[INFO] Reconstructing data from training.csv...")
    
    if not os.path.exists("training.csv"):
        print("[ERROR] training.csv not found. Cannot reconstruct training data.")
        return None, None, None, None
    
    df = pd.read_csv("training.csv")
    df = df.replace("NaN", np.nan)
    
    # Filter to only the columns we need (feature_names + Re-Tear for target)
    all_needed_cols = list(feature_names) + ['Re-Tear'] if 'Re-Tear' not in feature_names else list(feature_names)
    existing_cols = [c for c in all_needed_cols if c in df.columns]
    df = df[existing_cols].copy()
    
    # Create target
    if "Re-Tear" not in df.columns:
        print("[ERROR] 'Re-Tear' column not found in training.csv")
        return None, None, None, None
    
    y_raw = pd.to_numeric(df["Re-Tear"], errors="coerce")
    y_bin = (y_raw > 0).astype(int)
    valid_mask = ~y_raw.isna()
    df = df.loc[valid_mask].copy()
    y = y_bin.loc[valid_mask].copy()
    
    # Drop target column from features
    df = df.drop(columns=['Re-Tear'], errors='ignore')
    
    # Handle categorical columns
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).astype("category")
            if "Missing" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories("Missing")
            df[col] = df[col].fillna("Missing")
    
    # Handle numeric columns
    num_cols = [c for c in df.columns if c not in cat_cols]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Feature engineering (matching core.py)
    if feature_engineering:
        # Tear AP/ML Ratio
        if 'tear_characteristics_Tear_AntPost' in df.columns and 'tear_characteristics_Tear_MedLat' in df.columns:
            AP = df['tear_characteristics_Tear_AntPost']
            ML = df['tear_characteristics_Tear_MedLat']
            df['Tear_AP_ML_Ratio'] = np.where(ML > 0, AP / ML, np.nan)
            
            # Tear Area
            df['Tear_Area_cm2'] = (AP * ML) / 100
            df['Log_Tear_Area'] = np.log1p((AP * ML).clip(lower=0))
        
        # ER/IR Ratio
        if 'PREOP_Strength_IR' in df.columns and 'PREOP_Strength_ER' in df.columns:
            df['ER_IR_Ratio'] = np.where(
                df['PREOP_Strength_IR'] > 0,
                df['PREOP_Strength_ER'] / df['PREOP_Strength_IR'],
                np.nan
            )
        
        # ROM Deficit Score
        if all(c in df.columns for c in ['Pre-Op_ROM_Pre-Op_FF', 'Pre-Op_ROM_Pre-Op_Abd', 'Pre-Op_ROM_Pre-Op_ER']):
            df['ROM_Deficit_Score'] = (
                (180 - df['Pre-Op_ROM_Pre-Op_FF'].clip(upper=180)) / 180 +
                (180 - df['Pre-Op_ROM_Pre-Op_Abd'].clip(upper=180)) / 180 +
                (90 - df['Pre-Op_ROM_Pre-Op_ER'].clip(upper=90)) / 90
            ) / 3
        
        # Pain Frequency Mean
        if all(c in df.columns for c in ['PREOP_FOP_Activity_Pain', 'PREOP_FOP_Sleep_Pain', 'PREOP_FOP_Extreme_Pain']):
            df['Pain_Frequency_Mean'] = (
                df['PREOP_FOP_Activity_Pain'] +
                df['PREOP_FOP_Sleep_Pain'] +
                df['PREOP_FOP_Extreme_Pain']
            ) / 3
    
    # Ensure we only have the features that match feature_names
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"[WARN] Missing features after reconstruction: {missing_features}")
    
    # Filter to only feature_names columns that exist
    available_features = [f for f in feature_names if f in df.columns]
    X = df[available_features].copy()
    
    # Split with the exact same parameters as core.py
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    print(f"[INFO] Reconstructed - Train: {len(y_train)}, Test: {len(y_test)}")
    return X_train, y_train, X_test, y_test


def prepare_data_for_lr(X, cat_cols):
    """
    Prepare data for Logistic Regression:
    - One-hot encode categoricals
    - Handle missing values
    - Standardize numeric features
    """
    X_encoded = X.copy()
    
    # Handle categoricals: Clean strictly on X_encoded
    for col in cat_cols:
        if col in X_encoded.columns:
            # FIX: Modify X_encoded, not X
            X_encoded[col] = X_encoded[col].fillna('Missing').astype(str).str.upper()
    
    # One-hot encode
    if cat_cols:
        # Now get_dummies sees the clean, uppercased data
        X_encoded = pd.get_dummies(X_encoded, columns=[c for c in cat_cols if c in X_encoded.columns], 
                                   drop_first=True, dummy_na=False)
    
    # Fill numeric NaN with median
    for col in X_encoded.columns:
        if X_encoded[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())
    
    return X_encoded


def align_columns(X_train, X_test):
    """Ensure train and test have same columns."""
    # Get all columns
    all_cols = list(set(X_train.columns) | set(X_test.columns))
    
    # Add missing columns with zeros
    for col in all_cols:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Ensure same column order
    X_train = X_train[sorted(all_cols)]
    X_test = X_test[sorted(all_cols)]
    
    return X_train, X_test


# ==========================================
# FORMULA GENERATION
# ==========================================
def generate_formula(lr_model, feature_names):
    """
    Generate the full human-readable logistic regression formula 
    including ALL terms.
    
    P(retear) = 1 / (1 + exp(-z))
    where z = intercept + β1*x1 + β2*x2 + ...
    """
    coefs = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]
    
    # Create coefficient dataframe
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs),
        'Odds_Ratio': np.exp(coefs)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Start formula with intercept
    formula_parts = [f"{intercept:.4f}"]
    
    # Iterate through ALL rows in the dataframe (no slicing)
    for _, row in coef_df.iterrows():
        sign = "+" if row['Coefficient'] >= 0 else "-"
        # specific formatting for the terms
        formula_parts.append(f"{sign} {abs(row['Coefficient']):.4f} × {row['Feature']}")
    
    # Join all parts with newlines for readability
    formula_str = "z = " + "\n    ".join(formula_parts)
    
    formula_str += "\n\nP(retear) = 1 / (1 + exp(-z))"
    
    return formula_str, coef_df

# ==========================================
# MAIN
# ==========================================

def main(model_path, tearsense_probs=None):
    print("=" * 60)
    print("LOGISTIC REGRESSION BASELINE COMPARISON")
    print("=" * 60)
    
    # ─────────────────────────────────────────
    # 1. LOAD PKL
    # ─────────────────────────────────────────
    print(f"\n[INFO] Loading model bundle: {model_path}")
    pkg = joblib.load(model_path)
    
    serial_number = pkg.get('serial_number', 'unknown')
    output_dir = os.path.join("baseline_comparison", serial_number)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for training data
    X_train = pkg.get('X_train_all')
    y_train = pkg.get('y_train_all')
    X_test = pkg.get('X_test_exact')
    y_test = pkg.get('y_test_exact')
    cat_cols = pkg.get('cat_cols', [])
    feature_names = pkg.get('feature_names', [])
    
    # Get split ratio and feature engineering flag from pkg if available
    split_ratio = 0.2  # Default, can be overridden if stored
    feature_engineering = pkg.get('feature_engineering', True)
    
    if X_train is None or y_train is None:
        print("[INFO] Training data not in PKL. Attempting to reconstruct from training.csv...")
        
        # Try to reconstruct from training.csv
        X_train, y_train, X_test_reconstructed, y_test_reconstructed = reconstruct_data_from_csv(
            feature_names, cat_cols, 
            test_size=split_ratio, 
            feature_engineering=False
        )
        
        if X_train is None:
            print("[ERROR] Could not reconstruct training data.")
            print("[INFO] Make sure training.csv is in the current directory.")
            print("[INFO] Alternatively, update exporter.py to save training data, then re-export.")
            print("[INFO] Add these lines to the export_package dict in exporter.py:")
            print("       'X_train_all': holder.X_train_all,")
            print("       'y_train_all': holder.y_train_all,")
            return None
        
        # Use reconstructed X_test if the PKL doesn't have it
        if X_test is None:
            X_test = X_test_reconstructed
            y_test = y_test_reconstructed
            print("[INFO] Using reconstructed test data as well.")
        else:
            print("[INFO] Using test data from PKL, training data reconstructed from CSV.")
    
    # Convert to numpy if needed
    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    
    print(f"[INFO] Training samples: {len(y_train_arr)}")
    print(f"[INFO] Test samples: {len(y_test_arr)}")
    print(f"[INFO] Features: {len(feature_names)}")
    print(f"[INFO] Categorical columns: {len(cat_cols)}")
    
    # ─────────────────────────────────────────
    # 2. PREPARE DATA FOR LR
    # ─────────────────────────────────────────
    print("\n[INFO] Preparing data for Logistic Regression...")
    
    # Use only the feature columns that exist
    available_features = [f for f in feature_names if f in X_train.columns]
    if len(available_features) < len(feature_names):
        missing = set(feature_names) - set(available_features)
        print(f"[WARN] Missing {len(missing)} features: {list(missing)[:5]}...")
    
    X_train_subset = X_train[available_features].copy()
    X_test_subset = X_test[available_features].copy()
    
    # Encode and prepare
    X_train_enc = prepare_data_for_lr(X_train_subset, cat_cols)
    X_test_enc = prepare_data_for_lr(X_test_subset, cat_cols)
    
    # Align columns (one-hot encoding may produce different columns)
    X_train_enc, X_test_enc = align_columns(X_train_enc, X_test_enc)
    
    print(f"[INFO] Encoded features: {X_train_enc.shape[1]}")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)
    
    # ─────────────────────────────────────────
    # 3. TRAIN LOGISTIC REGRESSION
    # ─────────────────────────────────────────
    print("\n[INFO] Training Logistic Regression...")
    
    lr_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced data
    )
    
    lr_model.fit(X_train_scaled, y_train_arr)
    
    # Get predictions
    lr_probs_train = lr_model.predict_proba(X_train_scaled)[:, 1]
    lr_probs_test = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"[INFO] Training AUC: {roc_auc_score(y_train_arr, lr_probs_train):.4f}")
    
    # ─────────────────────────────────────────
    # 4. COMPUTE THRESHOLD (Youden's J on LR test probs)
    # ─────────────────────────────────────────
    lr_threshold = find_youden_threshold(y_test_arr, lr_probs_test)
    print(f"[INFO] LR Youden's J Threshold: {lr_threshold:.4f}")
    
    # Also get TearSense threshold for comparison
    ts_threshold = pkg.get('optimal_threshold', 0.5)
    
    # Use LR's own optimal threshold for LR metrics
    # This is the fairest comparison
    
    # ─────────────────────────────────────────
    # 5. COMPUTE METRICS
    # ─────────────────────────────────────────
    print("\n[INFO] Computing metrics...")
    
    lr_metrics = compute_all_metrics(y_test_arr, lr_probs_test, lr_threshold)
    lr_metrics['model'] = 'Logistic Regression'
    lr_metrics['n_features'] = int(X_train_enc.shape[1])
    lr_metrics['n_train'] = int(len(y_train_arr))
    lr_metrics['n_test'] = int(len(y_test_arr))
    
    # ─────────────────────────────────────────
    # 6. GENERATE FORMULA
    # ─────────────────────────────────────────
    print("\n[INFO] Generating formula...")
    
    formula_str, coef_df = generate_formula(lr_model, list(X_train_enc.columns))
    lr_metrics['formula'] = formula_str
    
    # Save coefficients
    coef_path = os.path.join(output_dir, "lr_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"[EXPORT] Coefficients saved to: {coef_path}")
    
    # ─────────────────────────────────────────
    # 7. LOAD TEARSENSE METRICS FOR COMPARISON
    # ─────────────────────────────────────────
    ts_metrics = pkg.get('metrics', {})
    ts_auc = ts_metrics.get('test_auc', 0)
    ts_brier = ts_metrics.get('test_brier', 0)
    
    # Reconstruct TearSense probs if possible (from external_assessor or provided)
    # For now, just use stored metrics
    
    # ─────────────────────────────────────────
    # 8. COMPARISON TABLE
    # ─────────────────────────────────────────
    comparison = {
        "serial_number": serial_number,
        "logistic_regression": lr_metrics,
        "tearsense": {
            "AUC": ts_auc,
            "Brier": ts_brier,
            "Threshold_Used": ts_threshold,
            "Note": "Full TearSense metrics available in external_assessor output"
        },
        "comparison": {
            "AUC_difference": ts_auc - lr_metrics['AUC'] if ts_auc else None,
            "Brier_difference": lr_metrics['Brier'] - ts_brier if ts_brier else None,
            "TearSense_better_AUC": ts_auc > lr_metrics['AUC'] if ts_auc else None
        }
    }
    
    # Save comparison
    comparison_path = os.path.join(output_dir, "comparison_table.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=4, cls=NumpyEncoder)
    
    # Save LR metrics
    lr_metrics_path = os.path.join(output_dir, "lr_metrics.json")
    with open(lr_metrics_path, 'w') as f:
        json.dump(lr_metrics, f, indent=4, cls=NumpyEncoder)
    print(f"[EXPORT] LR metrics saved to: {lr_metrics_path}")
    
    # ─────────────────────────────────────────
    # 9. PLOTS
    # ─────────────────────────────────────────
    print("\n[INFO] Generating plots...")
    
    # ROC Curve
    plt.figure(figsize=(8, 8))
    RocCurveDisplay.from_predictions(
        y_test_arr, lr_probs_test, 
        name=f"Logistic Regression (AUC={lr_metrics['AUC']:.3f})",
        color="blue"
    )
    if ts_auc:
        plt.title(f"ROC Comparison\nLR AUC: {lr_metrics['AUC']:.3f} | TearSense AUC: {ts_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "lr_roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calibration Curve
    plt.figure(figsize=(8, 8))
    CalibrationDisplay.from_predictions(
        y_test_arr, lr_probs_test,
        n_bins=10, name="Logistic Regression"
    )
    plt.title(f"Calibration Curve (Slope: {lr_metrics['Calibration_Slope']:.2f}, Brier: {lr_metrics['Brier']:.3f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "lr_calibration.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # DCA
    plt.figure(figsize=(10, 6))
    thresholds_dca = np.linspace(0.01, 0.99, 100)
    nb_lr = calculate_net_benefit_curve(y_test_arr, lr_probs_test, thresholds_dca)
    prevalence = np.mean(y_test_arr)
    nb_all = prevalence - (1 - prevalence) * (thresholds_dca / (1 - thresholds_dca))
    nb_none = np.zeros_like(thresholds_dca)
    
    plt.plot(thresholds_dca, nb_lr, color='blue', lw=2, label='Logistic Regression')
    plt.plot(thresholds_dca, nb_all, color='gray', linestyle='--', label='Treat All')
    plt.plot(thresholds_dca, nb_none, color='black', linestyle=':', label='Treat None')
    
    # Mark optimal threshold
    nb_at_opt = lr_metrics['Net_Benefit']
    plt.scatter(lr_threshold, nb_at_opt, color='blue', s=100, zorder=5,
                label=f"LR Threshold ({lr_threshold:.2f})")
    
    plt.ylim(-0.05, max(prevalence, max(nb_lr)) + 0.05)
    plt.xlim(0, 1.0)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis - Logistic Regression")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "lr_dca.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ─────────────────────────────────────────
    # 10. PRINT REPORT
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 60)
    print(f"\nModel: Logistic Regression (L2 regularization, class_weight='balanced')")
    print(f"Training N: {lr_metrics['n_train']}")
    print(f"Test N: {lr_metrics['n_test']}")
    print(f"Features: {lr_metrics['n_features']}")
    print(f"Threshold (Youden's J): {lr_threshold:.4f}")
    print(f"\n{'-'*60}")
    print(f"{'Metric':<25} | {'Value':>10}")
    print(f"{'-'*60}")
    print(f"{'AUC':<25} | {lr_metrics['AUC']:>10.4f}")
    print(f"{'Brier Score':<25} | {lr_metrics['Brier']:>10.4f}")
    print(f"{'Calibration Slope':<25} | {lr_metrics['Calibration_Slope']:>10.4f}")
    print(f"{'Net Benefit':<25} | {lr_metrics['Net_Benefit']:>10.4f}")
    print(f"{'Sensitivity':<25} | {lr_metrics['Sensitivity']:>10.4f}")
    print(f"{'Specificity':<25} | {lr_metrics['Specificity']:>10.4f}")
    print(f"{'PPV':<25} | {lr_metrics['PPV']:>10.4f}")
    print(f"{'NPV':<25} | {lr_metrics['NPV']:>10.4f}")
    print(f"{'F1 Score':<25} | {lr_metrics['F1_Score']:>10.4f}")
    print(f"{'-'*60}")
    
    cm = lr_metrics['Confusion_Matrix']
    print(f"\nConfusion Matrix: TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")
    
    # Print comparison if available
    if ts_auc:
        print(f"\n{'='*60}")
        print("COMPARISON: Logistic Regression vs TearSense")
        print(f"{'='*60}")
        print(f"{'Metric':<25} | {'LR':>10} | {'TearSense':>10} | {'Δ':>10}")
        print(f"{'-'*60}")
        print(f"{'AUC':<25} | {lr_metrics['AUC']:>10.4f} | {ts_auc:>10.4f} | {ts_auc - lr_metrics['AUC']:>+10.4f}")
        if ts_brier:
            print(f"{'Brier':<25} | {lr_metrics['Brier']:>10.4f} | {ts_brier:>10.4f} | {ts_brier - lr_metrics['Brier']:>+10.4f}")
        print(f"{'-'*60}")
        
        if ts_auc > lr_metrics['AUC']:
            print(f"\n✓ TearSense outperforms LR by {(ts_auc - lr_metrics['AUC'])*100:.2f}% AUC")
        else:
            print(f"\n✗ LR outperforms TearSense by {(lr_metrics['AUC'] - ts_auc)*100:.2f}% AUC")
    
    # Print formula
    print(f"\n{'='*60}")
    print("LOGISTIC REGRESSION FORMULA")
    print(f"{'='*60}")
    print(formula_str)
    
    print(f"\n{'='*60}")
    print(f"[EXPORT] All outputs saved to: {output_dir}/")
    print("[DONE]")
    
    return lr_metrics, comparison


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    # Option 1: Command line
    parser = argparse.ArgumentParser(description="LR baseline comparison for TearSense")
    parser.add_argument("--model_path", type=str, required=False, help="Path to .pkl file")
    args = parser.parse_args()
    
    if args.model_path:
        main(args.model_path)
    else:
        # Option 2: Direct path for testing
        serials_to_run = [
            '03022026_115404_58666',
        ]
        
        for serial in serials_to_run:
            model_path = f'outputs/{serial}/model/{serial}.pkl'
            if os.path.exists(model_path):
                main(model_path)
            else:
                print(f"[WARNING] Model not found: {model_path}")