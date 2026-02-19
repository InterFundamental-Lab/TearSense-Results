"""
TearSense + Logistic Regression Combiner
═════════════════════════════════════════════════════════════════════════════

Creates combined comparison plots for TearSense pipeline vs Logistic Regression.

LR training is handled by logistic_regressioner.py (single source of truth).

Pipeline Flow:
    TearSense:  raw input → feature engineering → 4 ensemble inference (CatBoost/XGBoost/LightGBM/RF)
                          → meta-features → meta-learner → calibration → output

    Logistic Regression:  raw input → one-hot encoding → standardization → LR → output
                          (trained via logistic_regressioner.train_lr)

Outputs:
    - combined_auroc.png        (ROC curves for both models + Youden thresholds)
    - combined_calibration.png  (Calibration curves for both models)
    - combined_dca.png          (DCA with treat all, treat none, both models + Youden thresholds)
    - combined_metrics.json     (Side-by-side metrics comparison)

Usage:
    python combiner.py --model_path outputs/SERIAL/model/SERIAL.pkl
"""

import argparse
import joblib
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT LR TRAINING FROM SINGLE SOURCE OF TRUTH
# ═══════════════════════════════════════════════════════════════════════════════
from runtime.externer_assessor_LogR import (
    train_lr,
    compute_all_metrics,
    find_youden_threshold,
    calculate_net_benefit,
    calculate_net_benefit_curve,
    get_train_test_from_bundle,
    reconstruct_data_from_csv,
    NumpyEncoder,
)


# ══════════════════════════════════════════════════════════════════════════════
# TEARSENSE INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_meta_features(base_preds, use_interactions=True, use_rank=False):
    """
    Reconstruct meta-features matching defecator.meta_features exactly.

    Features: [cat, xgb, lgbm, rf] + pairwise interactions + [mean, std, range]
    Total: 4 + 6 + 3 = 13 features
    """
    models = ['cat', 'xgb', 'lgbm', 'rf']

    clean = {}
    for m in models:
        c = np.nan_to_num(base_preds[m], nan=0.5)
        clean[m] = np.clip(c, 1e-7, 1 - 1e-7)

    X_meta = np.column_stack([clean[m] for m in models])

    if use_interactions:
        # Pairwise products
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i < j:
                    X_meta = np.column_stack([X_meta, clean[m1] * clean[m2]])

        # Summary statistics
        all_preds = np.array([clean[m] for m in models])
        X_meta = np.column_stack([
            X_meta,
            np.mean(all_preds, axis=0),
            np.std(all_preds, axis=0),
            np.max(all_preds, axis=0) - np.min(all_preds, axis=0),
        ])

    return X_meta


def run_tearsense_inference(bundle, X_test_raw, feature_names, cat_cols):
    """
    Run the full TearSense inference pipeline:

    raw input → 4 ensemble models (5-fold each) → meta-features → meta-learner → calibration → output
    """
    fold_models = bundle.get('fold_models')
    n_folds = bundle.get('n_folds', 5)

    if fold_models is None:
        raise ValueError("No fold_models in bundle")

    # Ensure X_test_raw is DataFrame with correct columns
    if not isinstance(X_test_raw, pd.DataFrame):
        X_test_raw = pd.DataFrame(X_test_raw, columns=feature_names)
    else:
        X_test_raw = pd.DataFrame(X_test_raw.values, columns=feature_names)

    # Create encoded version for XGB/LGBM/RF (matching core.py)
    X_test_xgb = bundle.get('X_test_xgb')
    if X_test_xgb is not None:
        if not isinstance(X_test_xgb, pd.DataFrame):
            X_test_enc = pd.DataFrame(X_test_xgb, columns=feature_names).fillna(-999)
        else:
            X_test_enc = pd.DataFrame(X_test_xgb.values, columns=feature_names).fillna(-999)
    else:
        X_test_enc = X_test_raw.copy()
        for col in cat_cols:
            if col in X_test_enc.columns:
                X_test_enc[col] = X_test_enc[col].astype('category').cat.codes
        X_test_enc = X_test_enc.fillna(-999)

    # ═══ Step 1: Base model predictions (4 algorithms × 5 folds) ═══
    print("[TearSense] Running 4-ensemble inference (CatBoost/XGBoost/LightGBM/RF)...")
    base_preds = {}

    for algo in ['cat', 'xgb', 'lgbm', 'rf']:
        algo_preds = np.zeros(len(X_test_raw))
        for model in fold_models[algo]:
            if algo == 'cat':
                preds = model.predict_proba(X_test_raw)[:, 1]
            else:
                try:
                    preds = model.predict_proba(X_test_enc)[:, 1]
                except ValueError:
                    # Fallback to numpy array
                    preds = model.predict_proba(X_test_enc.values)[:, 1]
            algo_preds += preds
        base_preds[algo] = algo_preds / n_folds
        print(f"    {algo.upper()}: mean={base_preds[algo].mean():.4f}")

    # ═══ Step 2: Meta-learner or weighted average ═══
    is_weighted_avg = bundle.get('is_weighted_avg', False)
    stacking_config = bundle.get('stacking_config', {})
    use_interactions = stacking_config.get('use_interactions', True)
    use_rank = stacking_config.get('use_rank_features', False)

    if is_weighted_avg:
        print("[TearSense] Combining via weighted average...")
        w = bundle.get('weights')
        if isinstance(w, dict):
            raw_probs = (w['cat'] * base_preds['cat'] + w['xgb'] * base_preds['xgb'] +
                        w['lgbm'] * base_preds['lgbm'] + w['rf'] * base_preds['rf'])
        else:
            w = bundle.get('weights_array')
            raw_probs = sum(w[i] * base_preds[m] for i, m in enumerate(['cat', 'xgb', 'lgbm', 'rf']))
    else:
        print("[TearSense] Generating meta-features and running meta-learner...")
        X_meta = generate_meta_features(base_preds, use_interactions=use_interactions, use_rank=use_rank)
        meta_model = bundle.get('meta_model')
        raw_probs = meta_model.predict_proba(X_meta)[:, 1]

    # ═══ Step 3: Calibration ═══
    calibrator = bundle.get('calibrator')
    if calibrator is not None:
        print("[TearSense] Applying calibration...")
        ts_probs = calibrator.predict(raw_probs)
    else:
        ts_probs = raw_probs

    return ts_probs, base_preds


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_combined_auroc(y_true, ts_probs, lr_probs, ts_auc, lr_auc,
                        ts_threshold, lr_threshold, output_path):
    """Create combined AUROC plot with both models and Youden threshold markers."""

    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate ROC curves
    fpr_ts, tpr_ts, _ = roc_curve(y_true, ts_probs)
    fpr_lr, tpr_lr, _ = roc_curve(y_true, lr_probs)

    # Plot ROC curves
    ax.plot(fpr_ts, tpr_ts, color='#E74C3C', lw=2.5,
            label=f'TearSense (AUC = {ts_auc:.3f})')
    ax.plot(fpr_lr, tpr_lr, color='#3498DB', lw=2.5,
            label=f'Logistic Regression (AUC = {lr_auc:.3f})')

    # Chance line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC = 0.500)')

    # Calculate operating points at Youden thresholds
    # TearSense
    ts_preds_at_thresh = (ts_probs >= ts_threshold).astype(int)
    ts_tpr = np.sum((ts_preds_at_thresh == 1) & (y_true == 1)) / np.sum(y_true == 1)
    ts_fpr = np.sum((ts_preds_at_thresh == 1) & (y_true == 0)) / np.sum(y_true == 0)

    # Logistic Regression
    lr_preds_at_thresh = (lr_probs >= lr_threshold).astype(int)
    lr_tpr = np.sum((lr_preds_at_thresh == 1) & (y_true == 1)) / np.sum(y_true == 1)
    lr_fpr = np.sum((lr_preds_at_thresh == 1) & (y_true == 0)) / np.sum(y_true == 0)

    # Plot threshold markers
    ax.scatter([ts_fpr], [ts_tpr], color='#E74C3C', s=200, zorder=5,
               edgecolors='black', linewidths=2, marker='o',
               label=f'TS Youden (t={ts_threshold:.2f})')
    ax.scatter([lr_fpr], [lr_tpr], color='#3498DB', s=200, zorder=5,
               edgecolors='black', linewidths=2, marker='s',
               label=f'LR Youden (t={lr_threshold:.2f})')

    # Styling
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    ax.set_title('ROC Curve Comparison\nTearSense vs Logistic Regression', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {output_path}")


def plot_combined_calibration(y_true, ts_probs, lr_probs, ts_brier, lr_brier, output_path):
    """Create combined calibration plot with both models."""

    fig, ax = plt.subplots(figsize=(10, 10))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')

    # TearSense calibration curve
    prob_true_ts, prob_pred_ts = calibration_curve(y_true, ts_probs, n_bins=10, strategy='uniform')
    ax.plot(prob_pred_ts, prob_true_ts, 's-', color='#E74C3C', lw=2.5, markersize=10,
            label=f'TearSense (Brier = {ts_brier:.3f})')

    # Logistic Regression calibration curve
    prob_true_lr, prob_pred_lr = calibration_curve(y_true, lr_probs, n_bins=10, strategy='uniform')
    ax.plot(prob_pred_lr, prob_true_lr, 'o-', color='#3498DB', lw=2.5, markersize=10,
            label=f'Logistic Regression (Brier = {lr_brier:.3f})')

    # Styling
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax.set_ylabel('Fraction of Positives (Observed)', fontsize=14)
    ax.set_title('Calibration Curve Comparison\nTearSense vs Logistic Regression', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {output_path}")


def plot_combined_dca(y_true, ts_probs, lr_probs, ts_threshold, lr_threshold, output_path):
    """
    Create combined Decision Curve Analysis plot with:
    - TearSense model curve + Youden threshold marker
    - Logistic Regression curve + Youden threshold marker
    - Treat All reference line
    - Treat None reference line
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    thresholds = np.linspace(0.01, 0.80, 100)  # Clinical range
    prevalence = float(np.mean(y_true))

    # Net benefit curves
    nb_ts = calculate_net_benefit_curve(y_true, ts_probs, thresholds)
    nb_lr = calculate_net_benefit_curve(y_true, lr_probs, thresholds)

    # Treat All: NB = prevalence - (1-prevalence) * (t / (1-t))
    nb_all = []
    for t in thresholds:
        if t < 1:
            nb_all.append(prevalence - (1 - prevalence) * (t / (1 - t)))
        else:
            nb_all.append(0)

    # Treat None: NB = 0
    nb_none = np.zeros_like(thresholds)

    # Plot model curves
    ax.plot(thresholds, nb_ts, color='#E74C3C', lw=3, label='TearSense')
    ax.plot(thresholds, nb_lr, color='#3498DB', lw=3, label='Logistic Regression')

    # Plot reference lines
    ax.plot(thresholds, nb_all, color='#95A5A6', lw=2, linestyle='--', label='Treat All')
    ax.plot(thresholds, nb_none, color='#2C3E50', lw=2, linestyle=':', label='Treat None')

    # Calculate and mark Youden threshold points
    ts_nb_at_youden = calculate_net_benefit(y_true, ts_probs, ts_threshold)
    lr_nb_at_youden = calculate_net_benefit(y_true, lr_probs, lr_threshold)

    # Vertical lines at thresholds
    ax.axvline(x=ts_threshold, color='#E74C3C', linestyle='--', alpha=0.5, lw=1.5)
    ax.axvline(x=lr_threshold, color='#3498DB', linestyle='--', alpha=0.5, lw=1.5)

    # Threshold markers
    ax.scatter([ts_threshold], [ts_nb_at_youden], color='#E74C3C', s=200, zorder=5,
               edgecolors='black', linewidths=2, marker='o',
               label=f'TS Youden (t={ts_threshold:.2f}, NB={ts_nb_at_youden:.3f})')
    ax.scatter([lr_threshold], [lr_nb_at_youden], color='#3498DB', s=200, zorder=5,
               edgecolors='black', linewidths=2, marker='s',
               label=f'LR Youden (t={lr_threshold:.2f}, NB={lr_nb_at_youden:.3f})')

    # Styling — clip y-axis to clinically relevant range
    # (Treat All plunges far negative; don't let it stretch the plot)
    y_min = max(min(min(nb_ts), min(nb_lr)), -0.1)
    y_max = max(max(nb_ts), max(nb_lr), prevalence) + 0.05
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, 0.80)
    ax.set_xlabel('Threshold Probability', fontsize=14)
    ax.set_ylabel('Net Benefit', fontsize=14)
    ax.set_title('Decision Curve Analysis\nTearSense vs Logistic Regression', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.4)

    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def main(model_path):
    print("=" * 75)
    print("  TEARSENSE + LOGISTIC REGRESSION COMBINER")
    print("=" * 75)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. LOAD MODEL BUNDLE
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading model bundle...")
    print(f"      Path: {model_path}")
    bundle = joblib.load(model_path)

    serial_number = bundle.get('serial_number', 'unknown')
    output_dir = os.path.join("combined_analysis", serial_number)
    os.makedirs(output_dir, exist_ok=True)

    # Extract data via shared helper
    X_train, y_train, X_test, y_test, cat_cols, feature_names, feature_engineering = \
        get_train_test_from_bundle(bundle)

    y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)

    print(f"      Serial: {serial_number}")
    print(f"      Test samples: {len(y_test_arr)}")
    print(f"      Features: {len(feature_names)}")
    print(f"      Prevalence: {np.mean(y_test_arr):.1%}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. RUN TEARSENSE INFERENCE PIPELINE
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[2/6] Running TearSense inference pipeline...")
    print("      raw → feature eng → 4 ensembles → meta-features → meta-learner → calibration → output")

    ts_probs, base_preds = run_tearsense_inference(bundle, X_test, feature_names, cat_cols)
    ts_threshold = find_youden_threshold(y_test_arr, ts_probs)
    ts_metrics = compute_all_metrics(y_test_arr, ts_probs, ts_threshold, "TearSense")

    print(f"\n      [TearSense Results]")
    print(f"      AUC:              {ts_metrics['AUC']:.4f}")
    print(f"      Brier:            {ts_metrics['Brier_Score']:.4f}")
    print(f"      Youden Threshold: {ts_threshold:.4f}")
    print(f"      Sensitivity:      {ts_metrics['Sensitivity_Recall']:.4f}")
    print(f"      Specificity:      {ts_metrics['Specificity']:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. TRAIN/RUN LOGISTIC REGRESSION (from logistic_regressioner.py)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[3/6] Training Logistic Regression baseline (via logistic_regressioner)...")

    # Ensure X_test is DataFrame
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    lr_result = train_lr(X_train, y_train_arr, X_test, cat_cols, feature_names)
    lr_probs = lr_result['probs_test']

    lr_threshold = find_youden_threshold(y_test_arr, lr_probs)
    lr_metrics = compute_all_metrics(y_test_arr, lr_probs, lr_threshold, "Logistic Regression")

    print(f"\n      [Logistic Regression Results]")
    print(f"      AUC:              {lr_metrics['AUC']:.4f}")
    print(f"      Brier:            {lr_metrics['Brier_Score']:.4f}")
    print(f"      Youden Threshold: {lr_threshold:.4f}")
    print(f"      Sensitivity:      {lr_metrics['Sensitivity_Recall']:.4f}")
    print(f"      Specificity:      {lr_metrics['Specificity']:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. GENERATE COMBINED PLOTS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[4/6] Generating combined AUROC plot...")
    plot_combined_auroc(
        y_test_arr, ts_probs, lr_probs,
        ts_metrics['AUC'], lr_metrics['AUC'],
        ts_threshold, lr_threshold,
        os.path.join(output_dir, "combined_auroc.png")
    )

    print(f"\n[5/6] Generating combined Calibration plot...")
    plot_combined_calibration(
        y_test_arr, ts_probs, lr_probs,
        ts_metrics['Brier_Score'], lr_metrics['Brier_Score'],
        os.path.join(output_dir, "combined_calibration.png")
    )

    print(f"\n[6/6] Generating combined DCA plot...")
    plot_combined_dca(
        y_test_arr, ts_probs, lr_probs,
        ts_threshold, lr_threshold,
        os.path.join(output_dir, "combined_dca.png")
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. SAVE COMBINED METRICS
    # ─────────────────────────────────────────────────────────────────────────
    combined_metrics = {
        "serial_number": serial_number,
        "n_test": int(len(y_test_arr)),
        "prevalence": float(np.mean(y_test_arr)),
        "tearsense": ts_metrics,
        "logistic_regression": lr_metrics,
        "comparison": {
            "AUC_difference": float(ts_metrics['AUC'] - lr_metrics['AUC']),
            "Brier_difference": float(ts_metrics['Brier_Score'] - lr_metrics['Brier_Score']),
            "TearSense_better_AUC": bool(ts_metrics['AUC'] > lr_metrics['AUC']),
            "TearSense_better_Brier": bool(ts_metrics['Brier_Score'] < lr_metrics['Brier_Score']),
            "AUC_relative_improvement_pct": float((ts_metrics['AUC'] - lr_metrics['AUC']) / lr_metrics['AUC'] * 100) if lr_metrics['AUC'] > 0 else 0.0,
        }
    }

    metrics_path = os.path.join(output_dir, "combined_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(combined_metrics, f, indent=4, cls=NumpyEncoder)
    print(f"\n[SAVED] {metrics_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. PRINT SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  COMPARISON SUMMARY")
    print("=" * 75)
    print(f"\n  {'Metric':<28} │ {'TearSense':>12} │ {'Logistic Reg':>12} │ {'Δ':>10}")
    print("  " + "─" * 71)
    print(f"  {'AUC':<28} │ {ts_metrics['AUC']:>12.4f} │ {lr_metrics['AUC']:>12.4f} │ {ts_metrics['AUC'] - lr_metrics['AUC']:>+10.4f}")
    print(f"  {'Brier Score':<28} │ {ts_metrics['Brier_Score']:>12.4f} │ {lr_metrics['Brier_Score']:>12.4f} │ {ts_metrics['Brier_Score'] - lr_metrics['Brier_Score']:>+10.4f}")
    print(f"  {'Calibration Slope':<28} │ {ts_metrics['Calibration_Slope']:>12.4f} │ {lr_metrics['Calibration_Slope']:>12.4f} │ {ts_metrics['Calibration_Slope'] - lr_metrics['Calibration_Slope']:>+10.4f}")
    print(f"  {'Net Benefit @ Youden':<28} │ {ts_metrics['Net_Benefit']:>12.4f} │ {lr_metrics['Net_Benefit']:>12.4f} │ {ts_metrics['Net_Benefit'] - lr_metrics['Net_Benefit']:>+10.4f}")
    print(f"  {'Sensitivity':<28} │ {ts_metrics['Sensitivity_Recall']:>12.4f} │ {lr_metrics['Sensitivity_Recall']:>12.4f} │ {ts_metrics['Sensitivity_Recall'] - lr_metrics['Sensitivity_Recall']:>+10.4f}")
    print(f"  {'Specificity':<28} │ {ts_metrics['Specificity']:>12.4f} │ {lr_metrics['Specificity']:>12.4f} │ {ts_metrics['Specificity'] - lr_metrics['Specificity']:>+10.4f}")
    print(f"  {'PPV':<28} │ {ts_metrics['PPV_Precision']:>12.4f} │ {lr_metrics['PPV_Precision']:>12.4f} │ {ts_metrics['PPV_Precision'] - lr_metrics['PPV_Precision']:>+10.4f}")
    print(f"  {'NPV':<28} │ {ts_metrics['NPV']:>12.4f} │ {lr_metrics['NPV']:>12.4f} │ {ts_metrics['NPV'] - lr_metrics['NPV']:>+10.4f}")
    print(f"  {'Youden Threshold':<28} │ {ts_threshold:>12.4f} │ {lr_threshold:>12.4f} │ {ts_threshold - lr_threshold:>+10.4f}")
    print("  " + "─" * 71)

    # Final verdict
    auc_diff = ts_metrics['AUC'] - lr_metrics['AUC']
    brier_diff = ts_metrics['Brier_Score'] - lr_metrics['Brier_Score']

    print(f"\n  VERDICT:")
    if auc_diff > 0:
        rel_improvement = abs(auc_diff) / lr_metrics['AUC'] * 100 if lr_metrics['AUC'] > 0 else 0
        print(f"  ✓ TearSense outperforms LR by {rel_improvement:.2f}% relative AUC improvement")
    else:
        rel_improvement = abs(auc_diff) / ts_metrics['AUC'] * 100 if ts_metrics['AUC'] > 0 else 0
        print(f"  ✗ LR outperforms TearSense by {rel_improvement:.2f}% relative AUC improvement")

    if brier_diff < 0:
        print(f"  ✓ TearSense has better calibration (lower Brier: {abs(brier_diff):.4f})")
    else:
        print(f"  ✗ LR has better calibration (lower Brier: {abs(brier_diff):.4f})")

    print(f"\n  All outputs saved to: {output_dir}/")
    print("=" * 75)

    return combined_metrics, ts_probs, lr_probs


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined TearSense + LR comparison analysis")
    parser.add_argument("--model_path", type=str, required=False,
                        help="Path to .pkl model file")
    args = parser.parse_args()

    if args.model_path:
        main(args.model_path)
    else:
        # Default for testing
        test_serials = ['03022026_115404_58666']
        for serial in test_serials:
            path = f'outputs/{serial}/model/{serial}.pkl'
            if os.path.exists(path):
                main(path)
            else:
                print(f"[WARN] Model not found: {path}")
                print("Usage: python combiner.py --model_path outputs/SERIAL/model/SERIAL.pkl")