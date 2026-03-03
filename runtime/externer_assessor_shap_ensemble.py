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
    ENGINEERED_FEATURES,
    apply_feature_engineering,
    generate_meta_features,
    NumpyEncoder,
    prepare_shap_data,
    plot_shap_summary,
    plot_shap_bar,
    export_feature_importance,
)

def build_ensemble_predict_fn(bundle):
    meta_model = bundle.get('meta_model')
    calibrator = bundle.get('calibrator')
    is_wa = bundle.get('is_weighted_avg', False)
    weights = bundle.get('weights')
    weights_arr = bundle.get('weights_array')
    sc = bundle.get('stacking_config', {})
    use_inter = sc.get('use_interactions', True)
    use_rank = sc.get('use_rank_features', False)

    def predict(X_meta_np):
        n = X_meta_np.shape[0]
        base_preds = {
            'cat': X_meta_np[:, 0],
            'xgb': X_meta_np[:, 1],
            'lgbm': X_meta_np[:, 2],
            'rf': X_meta_np[:, 3],
        }

        if is_wa:
            if isinstance(weights, dict):
                raw_probs = (
                    weights['cat']  * base_preds['cat']  +
                    weights['xgb']  * base_preds['xgb']  +
                    weights['lgbm'] * base_preds['lgbm'] +
                    weights['rf']   * base_preds['rf']
                )
            else:
                raw_probs = sum(
                    weights_arr[i] * base_preds[m]
                    for i, m in enumerate(['cat', 'xgb', 'lgbm', 'rf'])
                )
        else:
            X_meta_full = generate_meta_features(
                base_preds, use_interactions=use_inter, use_rank=use_rank)
            raw_probs = meta_model.predict_proba(X_meta_full)[:, 1]

        if calibrator is not None:
            final = calibrator.predict(raw_probs)
        else:
            final = raw_probs
        return final
    return predict

def get_base_predictions(X_raw_np, raw_feat, all_feat, cat_cols, cat_idx_full, decode_maps, bundle):
    n = X_raw_np.shape[0]
    df = pd.DataFrame(X_raw_np, columns=raw_feat)

    cat_max_code = {col: max(decode_maps[col].keys()) for col in decode_maps}
    for col in cat_cols:
        if col not in df.columns or col not in decode_maps:
            continue
        mx = cat_max_code[col]
        codes = df[col].round().clip(0, mx).astype(int)
        df[col] = codes.map(decode_maps[col]).fillna('Missing')

    feat_eng = bundle.get('feature_engineering', True)
    if feat_eng:
        df = apply_feature_engineering(df)

    for col in all_feat:
        if col not in df.columns:
            df[col] = np.nan
    X_full = df[all_feat].copy()

    X_enc = X_full.copy()
    for col in cat_cols:
        if col in X_enc.columns:
            X_enc[col] = X_enc[col].astype(str).astype('category').cat.codes
    X_enc = X_enc.fillna(-999)

    fold_models = bundle['fold_models']
    n_folds = bundle.get('n_folds', 5)

    base_preds = {}
    for algo in ['cat', 'xgb', 'lgbm', 'rf']:
        accum = np.zeros(n)
        for model in fold_models[algo]:
            if algo == 'cat':
                try:
                    from catboost import Pool
                    preds = model.predict_proba(Pool(X_full, cat_features=cat_idx_full))[:, 1]
                except Exception:
                    preds = model.predict_proba(X_full)[:, 1]
            else:
                try:
                    preds = model.predict_proba(X_enc)[:, 1]
                except ValueError:
                    try:
                        preds = model.predict_proba(X_enc.values)[:, 1]
                    except Exception:
                        preds = model.predict_proba(X_enc, validate_features=False)[:, 1]
            accum += preds
        base_preds[algo] = accum / n_folds

    return np.column_stack([base_preds['cat'], base_preds['xgb'], base_preds['lgbm'], base_preds['rf']])

def main(model_path, output_dir=None, background_k=50):
    print("\n" + "=" * 70)
    print("  SHAP ANALYSIS — Ensemble Layer")
    print("=" * 70)

    bundle = joblib.load(model_path)
    serial = bundle.get('serial_number', 'unknown')

    if output_dir is None:
        output_dir = os.path.join("external_assessor", serial, "shap_ensemble")
    os.makedirs(output_dir, exist_ok=True)

    (X_test_shap, X_bg_shap, raw_feat, all_feat,
     cat_cols, cat_idx_full, decode_maps, medians) = prepare_shap_data(bundle)

    print("[SHAP-Ensemble] Extracting base model predictions for test and background sets...")
    X_test_base = get_base_predictions(X_test_shap, raw_feat, all_feat, cat_cols, cat_idx_full, decode_maps, bundle)
    X_bg_base = get_base_predictions(X_bg_shap, raw_feat, all_feat, cat_cols, cat_idx_full, decode_maps, bundle)
    ensemble_feat_names = ['CatBoost', 'XGBoost', 'LightGBM', 'RandomForest']

    predict_fn = build_ensemble_predict_fn(bundle)

    print(f"[SHAP-Ensemble] Building background summary (kmeans k={background_k})...")
    background = shap.kmeans(X_bg_base, min(background_k, len(X_bg_base)))

    print("[SHAP-Ensemble] Initialising KernelExplainer (Ensemble Layer)...")
    explainer = shap.KernelExplainer(predict_fn, background)

    t0 = time.time()
    n_test = X_test_base.shape[0]
    print(f"[SHAP-Ensemble] Computing SHAP values for {n_test} test samples...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_test_base, silent=True)
    elapsed = time.time() - t0
    print(f"[SHAP-Ensemble] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print("[SHAP-Ensemble] Generating plots...")
    X_display = pd.DataFrame(X_test_base, columns=ensemble_feat_names)

    plot_shap_summary(
        shap_values, X_display, ensemble_feat_names,
        os.path.join(output_dir, f"shap_summary_{serial}.png"), serial,
    )
    plot_shap_bar(
        shap_values, ensemble_feat_names,
        os.path.join(output_dir, f"shap_bar_{serial}.png"), serial,
    )

    importance_df = export_feature_importance(
        shap_values, ensemble_feat_names,
        os.path.join(output_dir, f"feature_importance_{serial}.csv"),
    )

    sv_df = pd.DataFrame(shap_values, columns=ensemble_feat_names)
    sv_path = os.path.join(output_dir, f"shap_values_{serial}.csv")
    sv_df.to_csv(sv_path, index=False)
    print(f"[SHAP-Ensemble] Raw SHAP values → {sv_path}")

    # Metadata
    metadata = {
        "serial_number": serial,
        "n_test_samples": n_test,
        "is_weighted_avg": bundle.get('is_weighted_avg', False),
        "best_method": bundle.get('best_method', 'unknown'),
        "background_k": background_k,
        "elapsed_seconds": round(elapsed, 1),
        "top_features": importance_df[['Rank', 'Feature', 'Mean_Abs_SHAP']].to_dict('records'),
    }
    meta_path = os.path.join(output_dir, f"shap_metadata_{serial}.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
    print(f"[SHAP-Ensemble] Metadata → {meta_path}")

    print("\n" + "=" * 70)
    print(f"  SHAP ANALYSIS COMPLETE — Ensemble Layer — {serial}")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        mp = sys.argv[1]
        od = sys.argv[2] if len(sys.argv) >= 3 else None
        main(mp, output_dir=od)
    else:
        print("Usage: python -m runtime.externer_assessor_shap_ensemble <model.pkl> [output_dir]")
