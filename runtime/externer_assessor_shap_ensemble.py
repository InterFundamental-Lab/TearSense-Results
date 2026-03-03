import os
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from runtime.externer_assessor_shap import (
    NumpyEncoder,
    plot_shap_summary,
    plot_shap_bar,
    export_feature_importance,
)

def main(model_path, output_dir=None):
    print("\n" + "=" * 70)
    print("  SHAP ANALYSIS — Weighted Ensemble Feature Importance")
    print("=" * 70)

    bundle = joblib.load(model_path)
    serial = bundle.get('serial_number', 'unknown')

    if output_dir is None:
        output_dir = os.path.join("external_assessor", serial, "shap_ensemble")
    os.makedirs(output_dir, exist_ok=True)

    all_feat = list(bundle.get('feature_names', []))
    cat_cols = list(bundle.get('cat_cols', []))
    cat_idx_full = list(bundle.get('cat_indices', []))

    # Read the exact weights from best_hyperparameters.json
    config_path = os.path.join("outputs", serial, "configs", "best_hyperparameters.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            hyperparams = json.load(f)
        ew = hyperparams.get("Ensemble_Weights", {})
        w = {
            'cat': ew.get('Cat', 0.0),
            'xgb': ew.get('XGB', 0.0),
            'lgbm': ew.get('LGBM', 0.0),
            'rf': ew.get('RF', 0.0)
        }
        print(f"[SHAP-Ensemble] Loaded weights from {config_path}: {w}")
    else:
        print(f"[WARN] Config not found at {config_path}. Falling back to equal weights.")
        w = {'cat': 0.25, 'xgb': 0.25, 'lgbm': 0.25, 'rf': 0.25}

    # Data Preparation
    X_test_raw = bundle.get('X_test_exact')
    if not isinstance(X_test_raw, pd.DataFrame):
        X_test_raw = pd.DataFrame(X_test_raw, columns=all_feat)
    else:
        X_test_raw = pd.DataFrame(X_test_raw.values, columns=all_feat)

    # For CatBoost (expects original strings/NaNS for categoricals)
    X_full = X_test_raw.copy()
    for col in cat_cols:
        if col in X_full.columns:
            X_full[col] = X_full[col].astype(str).replace('nan', 'Missing').fillna('Missing')

    # For XGBoost, LightGBM, RF (expects encoded numbers or matching training formats)
    X_enc = X_full.copy()
    for col in cat_cols:
        if col in X_enc.columns:
            X_enc[col] = X_enc[col].astype('category').cat.codes
    # RF doesn't like NaNs sometimes, XGB/LGBM handles them
    X_rf = X_enc.copy().fillna(-999)
    X_xgb_lgbm = X_enc.copy()

    shap_vals_dict = {}
    t0 = time.time()

    # Compute TreeSHAP for each base model by averaging the folds
    for algo in ['cat', 'xgb', 'lgbm', 'rf']:
        if w[algo] == 0:
            print(f"[SHAP-Ensemble] Skipping {algo} (weight is 0.0)")
            shap_vals_dict[algo] = np.zeros(X_enc.shape)
            continue
            
        print(f"[SHAP-Ensemble] Computing TreeSHAP for {algo} base models...")
        models = bundle.get('fold_models', {}).get(algo, [])
        if not models:
            print(f"   [WARN] No models found for {algo}.")
            shap_vals_dict[algo] = np.zeros(X_enc.shape)
            continue
            
        fold_shaps = []
        for i, model in enumerate(models):
            if algo == 'cat':
                from catboost import Pool
                pool = Pool(X_full, cat_features=cat_idx_full)
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(pool)
                # CatBoost explainer returns (N, F)
                fold_shaps.append(sv)
            elif algo == 'xgb':
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_xgb_lgbm)
                fold_shaps.append(sv)
            elif algo == 'lgbm':
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_xgb_lgbm)
                if isinstance(sv, list): sv = sv[1]
                fold_shaps.append(sv)
            elif algo == 'rf':
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_rf, check_additivity=False)
                if isinstance(sv, list): sv = sv[1]
                elif len(sv.shape) == 3: sv = sv[:, :, 1]
                fold_shaps.append(sv)
        
        shap_vals_dict[algo] = np.mean(fold_shaps, axis=0)

    print("\n -> Combining SHAP Values...")
    ensemble_shap = (
        (shap_vals_dict['cat'] * w['cat']) +
        (shap_vals_dict['xgb'] * w['xgb']) +
        (shap_vals_dict['lgbm'] * w['lgbm']) +
        (shap_vals_dict['rf'] * w['rf'])
    )

    elapsed = time.time() - t0
    print(f"[SHAP-Ensemble] Done in {elapsed:.1f}s")

    print("[SHAP-Ensemble] Generating plots...")
    # Convert categorical strings to numeric codes for SHAP coloring
    X_display = X_test_raw.copy()
    for col in cat_cols:
        if col in X_display.columns:
            X_display[col] = X_display[col].astype('category').cat.codes
            
    # Exclusion Criteria
    exclude_features = ['Logit_Retear_Risk']
    indices_to_keep = [i for i, f in enumerate(all_feat) if f not in exclude_features]
    filtered_feat = [all_feat[i] for i in indices_to_keep]
    
    ensemble_shap_filtered = ensemble_shap[:, indices_to_keep]
    X_display_filtered = X_display.iloc[:, indices_to_keep]

    plot_shap_summary(
        ensemble_shap_filtered, X_display_filtered, filtered_feat,
        os.path.join(output_dir, f"shap_summary_{serial}.png"), serial,
    )
    plot_shap_bar(
        ensemble_shap_filtered, filtered_feat,
        os.path.join(output_dir, f"shap_bar_{serial}.png"), serial,
    )

    importance_df = export_feature_importance(
        ensemble_shap_filtered, filtered_feat,
        os.path.join(output_dir, f"feature_importance_{serial}.csv"),
    )

    sv_df = pd.DataFrame(ensemble_shap_filtered, columns=filtered_feat)
    sv_path = os.path.join(output_dir, f"shap_values_{serial}.csv")
    sv_df.to_csv(sv_path, index=False)
    print(f"[SHAP-Ensemble] Raw SHAP values → {sv_path}")

    # Metadata
    metadata = {
        "serial_number": serial,
        "n_test_samples": len(X_test_raw),
        "weights_used": w,
        "elapsed_seconds": round(elapsed, 1),
        "top_features": importance_df.head(15)[['Rank', 'Feature', 'Mean_Abs_SHAP']].to_dict('records'),
    }
    meta_path = os.path.join(output_dir, f"shap_metadata_{serial}.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
    print(f"[SHAP-Ensemble] Metadata → {meta_path}")

    print("\n" + "=" * 70)
    print(f"  SHAP ANALYSIS COMPLETE — Weighted Ensemble — {serial}")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        mp = sys.argv[1]
        od = sys.argv[2] if len(sys.argv) >= 3 else None
        main(mp, output_dir=od)
    else:
        print("Usage: python -m runtime.externer_assessor_shap_ensemble <model.pkl> [output_dir]")