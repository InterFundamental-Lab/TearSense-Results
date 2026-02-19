import joblib
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, confusion_matrix,
    roc_curve, f1_score, RocCurveDisplay
)
from sklearn.calibration import CalibrationDisplay, calibration_curve

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT LR TRAINING + SHARED UTILITIES FROM SINGLE SOURCE OF TRUTH
# ═══════════════════════════════════════════════════════════════════════════════
from runtime.externer_assessor_LogR import (
    train_lr,
    compute_all_metrics,
    find_youden_threshold,
    calculate_net_benefit,
    calculate_net_benefit_curve,
    get_calibration_slope_intercept,
    bootstrap_metrics,
    generate_formula,
    NumpyEncoder,
)


# ══════════════════════════════════════════════════════════════════════════════
# TEARSENSE-SPECIFIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL BASE MODEL METRICS (kept here — not LR-related)
# ══════════════════════════════════════════════════════════════════════════════

def compute_individual_model_metrics(y_test, base_preds):
    """Evaluate each base model (CatBoost, XGBoost, LightGBM, RF) individually."""
    algo_display = {'cat': 'CatBoost', 'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'rf': 'RandomForest'}
    individual_model_metrics = {}

    print(f"\n{'='*105}")
    print("LAYER 1: INDIVIDUAL BASE MODEL EVALUATION (Holdout Test Set, Fold-Averaged)")
    print(f"{'='*105}")
    print(f"{'Model':<15} {'AUROC':>8} {'Cal.Slope':>10} {'Cal.Int':>10} {'Brier':>8} {'NetBen':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'NPV':>8}")
    print("-" * 105)

    for algo_key, display_name in algo_display.items():
        probs_i = base_preds[algo_key]
        thresh_i = find_youden_threshold(y_test, probs_i)
        preds_i = (probs_i >= thresh_i).astype(int)
        tn_i, fp_i, fn_i, tp_i = confusion_matrix(y_test, preds_i).ravel()

        auc_i = roc_auc_score(y_test, probs_i)
        brier_i = brier_score_loss(y_test, probs_i)
        cal_slope_i, cal_int_i = get_calibration_slope_intercept(y_test, probs_i)
        nb_i = calculate_net_benefit(y_test, probs_i, thresh_i)
        sens_i = float(tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else 0.0
        spec_i = float(tn_i / (tn_i + fp_i)) if (tn_i + fp_i) > 0 else 0.0
        ppv_i = float(tp_i / (tp_i + fp_i)) if (tp_i + fp_i) > 0 else 0.0
        npv_i = float(tn_i / (tn_i + fn_i)) if (tn_i + fn_i) > 0 else 0.0
        f1_i = float(f1_score(y_test, preds_i, zero_division=0))

        individual_model_metrics[display_name] = {
            "AUROC": auc_i,
            "Calibration_Slope": cal_slope_i,
            "Calibration_Intercept": cal_int_i,
            "Brier": brier_i,
            "Net_Benefit": nb_i,
            "Sensitivity": sens_i,
            "Specificity": spec_i,
            "PPV": ppv_i,
            "NPV": npv_i,
            "F1": f1_i,
            "Threshold_Used": float(thresh_i)
        }

        print(f"{display_name:<15} {auc_i:>8.4f} {cal_slope_i:>10.4f} {cal_int_i:>10.4f} {brier_i:>8.4f} {nb_i:>8.4f} {sens_i:>8.4f} {spec_i:>8.4f} {ppv_i:>8.4f} {npv_i:>8.4f}")

    print(f"{'='*105}")

    return individual_model_metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(model_path):
    bundle = load_model_bundle(model_path)
    serial_number = bundle.get('serial_number', 'unknown_serial')

    output_dir = os.path.join("external_assessor", serial_number)
    output_dir_auc = os.path.join("external_assessor", "auroc")
    output_dir_calibration = os.path.join("external_assessor", "calibration")
    output_dir_dca = os.path.join("external_assessor", "dca")
    output_dir_metrics = os.path.join("external_assessor", "metrics")

    for d in [output_dir, output_dir_auc, output_dir_calibration, output_dir_dca, output_dir_metrics]:
        os.makedirs(d, exist_ok=True)

    print(f"[INFO] Output Directory: {output_dir}")

    # ── EXTRACT DATA ──
    X_test_raw = bundle.get('X_test_exact')
    y_test = bundle.get('y_test_exact')
    fold_models = bundle.get('fold_models')
    n_folds = bundle.get('n_folds', 5)
    cat_cols = bundle.get('cat_cols', [])
    feature_names = bundle.get('feature_names', [])

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

    if not feature_names or len(feature_names) == 0:
        if isinstance(X_test_raw, pd.DataFrame):
            feature_names = list(X_test_raw.columns)
        else:
            raise ValueError("feature_names is empty and X_test_raw has no columns!")

    if fold_models.get('xgb') and len(fold_models['xgb']) > 0:
        try:
            xgb_feature_names = fold_models['xgb'][0].get_booster().feature_names
            if xgb_feature_names:
                feature_names = list(xgb_feature_names)
        except Exception as e:
            print(f"[DEBUG] Could not get feature names from XGBoost: {e}")

    print(f"\n[INFO] Running TearSense inference on {len(y_test)} patients...")

    if not isinstance(X_test_raw, pd.DataFrame):
        X_test_raw = pd.DataFrame(X_test_raw, columns=feature_names)
    else:
        X_test_raw.columns = feature_names

    X_test_xgb = bundle.get('X_test_xgb')
    if X_test_xgb is not None:
        if not isinstance(X_test_xgb, pd.DataFrame):
            X_test_enc = pd.DataFrame(X_test_xgb, columns=feature_names).fillna(-999)
        else:
            X_test_xgb.columns = feature_names
            X_test_enc = X_test_xgb.fillna(-999)
    else:
        X_test_enc = X_test_raw.copy()
        for col in cat_cols:
            if col in X_test_enc.columns:
                X_test_enc[col] = X_test_enc[col].astype('category').cat.codes
        X_test_enc = X_test_enc.fillna(-999)

    X_test_enc.columns = feature_names

    # ── BASE MODEL PREDICTIONS ──
    base_preds = {}

    for algo in ['cat', 'xgb', 'lgbm', 'rf']:
        algo_preds = np.zeros(len(y_test))
        for model in fold_models[algo]:
            if algo == 'cat':
                preds = model.predict_proba(X_test_raw)[:, 1]
            else:
                preds = None
                try:
                    preds = model.predict_proba(X_test_enc)[:, 1]
                except ValueError as e:
                    if "feature names" not in str(e).lower():
                        raise

                if preds is None and algo == 'xgb':
                    try:
                        preds = model.predict_proba(X_test_enc, validate_features=False)[:, 1]
                    except (ValueError, TypeError):
                        pass

                if preds is None and algo == 'xgb':
                    try:
                        import xgboost as xgb
                        dmatrix = xgb.DMatrix(X_test_enc.values, feature_names=feature_names)
                        raw_preds = model.get_booster().predict(dmatrix)
                        preds = raw_preds if raw_preds.max() <= 1 else 1 / (1 + np.exp(-raw_preds))
                    except Exception:
                        pass

                if preds is None:
                    preds = model.predict_proba(X_test_enc.values)[:, 1]

            algo_preds += preds
        base_preds[algo] = algo_preds / n_folds

    # ── LAYER 1: INDIVIDUAL BASE MODEL EVALUATION ──
    individual_model_metrics = compute_individual_model_metrics(y_test, base_preds)

    indiv_json_path = os.path.join(output_dir, f"individual_model_metrics_{serial_number}.json")
    with open(indiv_json_path, 'w') as f:
        json.dump(individual_model_metrics, f, indent=4, cls=NumpyEncoder)
    print(f"[EXPORT] Individual model metrics saved to {indiv_json_path}")

    indiv_csv_rows = []
    for name, m in individual_model_metrics.items():
        row = {'Model': name}
        row.update(m)
        indiv_csv_rows.append(row)
    indiv_csv_path = os.path.join(output_dir, f"individual_model_metrics_{serial_number}.csv")
    pd.DataFrame(indiv_csv_rows).to_csv(indiv_csv_path, index=False)

    # ── TEARSENSE COMBINED INFERENCE ──
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

    # ── LOGISTIC REGRESSION BASELINE (via logistic_regressioner.py) ──
    lr_probs = None
    lr_metrics = None
    lr_formula = None
    lr_coef_df = None

    if has_train_data:
        print(f"\n[INFO] Training Logistic Regression baseline (via logistic_regressioner)...")

        lr_result = train_lr(X_train, y_train_arr, X_test_raw, cat_cols, feature_names)

        lr_probs = lr_result['probs_test']
        lr_coef_df = lr_result['coef_df']
        lr_formula = lr_result['formula']

        lr_threshold = find_youden_threshold(y_test, lr_probs)
        lr_metrics = compute_all_metrics(y_test, lr_probs, lr_threshold, "Logistic Regression")

        print("[INFO] Bootstrapping LR confidence intervals...")
        lr_cis = bootstrap_metrics(y_test, lr_probs, lr_threshold, n_bootstraps=1000)
        for k, v in lr_cis.items():
            lr_metrics[f"{k}_95CI"] = [float(v[0]), float(v[1])]

        coef_path = os.path.join(output_dir, f"lr_coefficients_{serial_number}.csv")
        lr_coef_df.to_csv(coef_path, index=False)
        print(f"[EXPORT] LR coefficients saved to: {coef_path}")

    # ── TEARSENSE METRICS ──
    youden_thresh = find_youden_threshold(y_test, ts_probs)
    preds_binary = (ts_probs >= youden_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_binary).ravel()
    cal_slope, cal_intercept = get_calibration_slope_intercept(y_test, ts_probs)
    net_benefit = calculate_net_benefit(y_test, ts_probs, youden_thresh)

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
        "Sensitivity_Recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "PPV_Precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "NPV": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        "Confusion_Matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
    }

    print("[INFO] Bootstrapping TearSense confidence intervals...")
    ts_cis = bootstrap_metrics(y_test, ts_probs, youden_thresh, n_bootstraps=1000)

    for k, v in ts_cis.items():
        ts_metrics[f"{k}_95CI"] = [float(v[0]), float(v[1])]
        point_est = ts_metrics.get(k, 0.0)
        ts_metrics[f"{k}_Text"] = f"{point_est:.4f} ({v[0]:.4f}-{v[1]:.4f})"

    combined_metrics = {
        "tearsense": ts_metrics,
        "individual_base_models": individual_model_metrics,
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

    json_path = os.path.join(output_dir, f"clinical_metrics_{serial_number}.json")
    with open(json_path, 'w') as f:
        json.dump(combined_metrics, f, indent=4, cls=NumpyEncoder)
    print(f"[EXPORT] Metrics saved to {json_path}")

    json_path_combined = os.path.join(output_dir_metrics, f"{serial_number}.json")
    with open(json_path_combined, 'w') as f:
        json.dump(combined_metrics, f, indent=4, cls=NumpyEncoder)

    # ── PRINT REPORT ──
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

    # ── PLOTS ──
    print("\n[INFO] Generating comparison plots...")

    fig, ax = plt.subplots(figsize=(8, 8))

    RocCurveDisplay.from_predictions(
        y_test, ts_probs,
        name=f"TearSense (AUC={ts_metrics['AUC']:.3f})",
        color="darkorange", ax=ax
    )

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

    fig, ax = plt.subplots(figsize=(8, 8))

    CalibrationDisplay.from_predictions(y_test, ts_probs, n_bins=10, name="TearSense", ax=ax)
    if lr_probs is not None:
        CalibrationDisplay.from_predictions(y_test, lr_probs, n_bins=10, name="Logistic Regression", ax=ax)

    ax.set_title(f"Calibration Comparison — {serial_number}")
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(os.path.join(output_dir, f"calibration_comparison_{serial_number}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir_calibration, f"calibration_comparison_{serial_number}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds_dca = np.linspace(0.01, 0.99, 100)

    nb_ts = calculate_net_benefit_curve(y_test, ts_probs, thresholds_dca)
    ax.plot(thresholds_dca, nb_ts, color='darkorange', lw=2, label='TearSense')

    if lr_probs is not None:
        nb_lr = calculate_net_benefit_curve(y_test, lr_probs, thresholds_dca)
        ax.plot(thresholds_dca, nb_lr, color='blue', lw=2, label='Logistic Regression')

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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python external_assessor.py <path_to_model_bundle.pkl>")
