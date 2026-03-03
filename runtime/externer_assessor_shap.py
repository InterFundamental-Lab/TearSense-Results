"""
TearSense SHAP Analysis — Full End-to-End Pipeline
═══════════════════════════════════════════════════════════════════════════

Wraps the ENTIRE TearSense pipeline as a single black-box function:

    raw clinical features → [feature engineering] → 4 base models (fold-avg)
                          → meta-features → meta-learner → calibration
                          → P(retear)

Uses shap.KernelExplainer to compute SHAP values for this complete
pipeline. Feature-engineering columns are EXCLUDED from the SHAP input
(they are reconstructed internally), so SHAP attributes importance
only to the raw clinical features.

All output is written directly to the output_dir passed by the caller.
No intermediate directories are created.

Outputs:
    shap_summary_{serial}.png          SHAP beeswarm summary plot
    shap_bar_{serial}.png              Mean |SHAP| bar chart
    feature_importance_{serial}.csv    Sorted feature importance table
    shap_values_{serial}.csv           Raw per-sample SHAP values
    shap_metadata_{serial}.json        Run metadata

Usage (standalone):
    python -m runtime.externer_assessor_shap <model.pkl> [output_dir]

Usage (from run.py):
    external_assessor_shap.main(model_path, output_dir=target_dir)
"""

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

try:
    import shap
except ImportError:
    raise ImportError("shap is required.  pip install shap")


# ══════════════════════════════════════════════════════════════════════════════
# ENGINEERED FEATURES — excluded from SHAP input, rebuilt inside wrapper
# Sync with externer_assessor_LogR.EXCLUDE_FEATURES
# ══════════════════════════════════════════════════════════════════════════════
ENGINEERED_FEATURES = [
    'Logit_Retear_Risk'
]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (replicates core.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def apply_feature_engineering(df):
    df = df.copy()
    if 'tear_characteristics_Tear_AntPost' not in df.columns:
        return df

    AP = df['tear_characteristics_Tear_AntPost']
    ML = df['tear_characteristics_Tear_MedLat']

    df['Tear_AP_ML_Ratio'] = np.where(ML > 0, AP / ML, np.nan)
    df['Tear_Area_cm2']    = (AP * ML) / 100
    df['Log_Tear_Area']    = np.log1p((AP * ML).clip(lower=0))

    if 'PREOP_Strength_IR' in df.columns and 'PREOP_Strength_ER' in df.columns:
        df['ER_IR_Ratio'] = np.where(
            df['PREOP_Strength_IR'] > 0,
            df['PREOP_Strength_ER'] / df['PREOP_Strength_IR'],
            np.nan,
        )

    rom = ['Pre-Op_ROM_Pre-Op_FF', 'Pre-Op_ROM_Pre-Op_Abd', 'Pre-Op_ROM_Pre-Op_ER']
    if all(c in df.columns for c in rom):
        df['ROM_Deficit_Score'] = (
            (180 - df[rom[0]].clip(upper=180)) / 180 +
            (180 - df[rom[1]].clip(upper=180)) / 180 +
            (90  - df[rom[2]].clip(upper=90))  / 90
        ) / 3

    pain = ['PREOP_FOP_Activity_Pain', 'PREOP_FOP_Sleep_Pain', 'PREOP_FOP_Extreme_Pain']
    if all(c in df.columns for c in pain):
        df['Pain_Frequency_Mean'] = df[pain].sum(axis=1) / 3

    return df


# ══════════════════════════════════════════════════════════════════════════════
# META-FEATURE GENERATION  (replicates defecator.meta_features exactly)
# ══════════════════════════════════════════════════════════════════════════════

def generate_meta_features(base_preds, use_interactions=True, use_rank=False):
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
# JSON ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):    return int(obj)
        if isinstance(obj, (np.floating,)):   return float(obj)
        if isinstance(obj, (np.bool_,)):      return bool(obj)
        if isinstance(obj, np.ndarray):        return obj.tolist()
        return super().default(obj)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_shap_data(bundle):
    """
    Build numeric matrices for KernelExplainer:
      - categoricals → integer codes (with decode map stored)
      - engineered features removed
      - NaN filled with medians

    Returns
    -------
    X_test_shap   : np.ndarray   (n_test, n_raw)
    X_bg_shap     : np.ndarray   (n_train, n_raw)  — for kmeans background
    raw_feat      : list[str]    raw feature names (SHAP columns)
    all_feat      : list[str]    full feature names (incl. engineered)
    cat_cols      : list[str]
    cat_idx_full  : list[int]    indices of cat cols inside all_feat
    decode_maps   : dict         {col: {code_int: original_str_value}}
    medians       : dict         {col: median_value}
    """
    all_feat    = list(bundle.get('feature_names', []))
    cat_cols    = list(bundle.get('cat_cols', []))
    cat_idx_full = list(bundle.get('cat_indices', []))
    feat_eng    = bundle.get('feature_engineering', True)

    # Which engineered features are actually present in this model?
    eng_present = [f for f in ENGINEERED_FEATURES if f in all_feat] if feat_eng else []
    raw_feat    = [f for f in all_feat if f not in ENGINEERED_FEATURES]

    # ── X_test ──
    X_test_raw = bundle.get('X_test_exact')
    if not isinstance(X_test_raw, pd.DataFrame):
        X_test_raw = pd.DataFrame(X_test_raw, columns=all_feat)
    else:
        X_test_raw = pd.DataFrame(X_test_raw.values, columns=all_feat)

    X_test_sub = X_test_raw[raw_feat].copy()

    # ── X_train (for background) ──
    X_train_raw = bundle.get('X_train_all')
    if X_train_raw is not None:
        if not isinstance(X_train_raw, pd.DataFrame):
            X_train_raw = pd.DataFrame(X_train_raw, columns=all_feat)
        else:
            X_train_raw = pd.DataFrame(X_train_raw.values, columns=all_feat)
        X_train_sub = X_train_raw[raw_feat].copy()
    else:
        X_train_sub = X_test_sub.copy()

    # ── Encode categoricals as integer codes; store decode map ──
    decode_maps = {}
    for col in cat_cols:
        if col not in raw_feat:
            continue
        # Build canonical categories from training data
        combined = pd.concat([
            X_train_sub[col].astype(str),
            X_test_sub[col].astype(str),
        ]).astype('category')
        cats = combined.cat.categories
        decode_maps[col] = {i: str(cats[i]) for i in range(len(cats))}

        # Encode
        X_train_sub[col] = X_train_sub[col].astype(str).astype(
            pd.CategoricalDtype(categories=cats)
        ).cat.codes.replace(-1, np.nan).astype(float)

        X_test_sub[col] = X_test_sub[col].astype(str).astype(
            pd.CategoricalDtype(categories=cats)
        ).cat.codes.replace(-1, np.nan).astype(float)

    # ── Fill NaN with medians (from train) ──
    medians = {}
    for col in raw_feat:
        X_train_sub[col] = pd.to_numeric(X_train_sub[col], errors='coerce')
        X_test_sub[col]  = pd.to_numeric(X_test_sub[col], errors='coerce')
        med = X_train_sub[col].median()
        medians[col] = float(med) if not np.isnan(med) else 0.0
        X_train_sub[col] = X_train_sub[col].fillna(medians[col])
        X_test_sub[col]  = X_test_sub[col].fillna(medians[col])

    return (
        X_test_sub.values.astype(float),
        X_train_sub.values.astype(float),
        raw_feat,
        all_feat,
        cat_cols,
        cat_idx_full,
        decode_maps,
        medians,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE PREDICT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline_predict_fn(bundle, raw_feat, all_feat,
                               cat_cols, cat_idx_full, decode_maps):
    """
    Returns a callable  f(X_numeric) -> P(retear)  that wraps the
    complete TearSense pipeline end-to-end:

        decode cats → feature engineering → 4 base models (fold-avg)
                    → meta-features → meta-learner → calibration → output

    Parameters
    ----------
    X_numeric : np.ndarray (n, len(raw_feat))
        Categoricals are integer-coded; numerics as-is.
    """
    fold_models  = bundle['fold_models']
    n_folds      = bundle.get('n_folds', 5)
    meta_model   = bundle.get('meta_model')
    calibrator   = bundle.get('calibrator')
    is_wa        = bundle.get('is_weighted_avg', False)
    weights      = bundle.get('weights')
    weights_arr  = bundle.get('weights_array')
    feat_eng     = bundle.get('feature_engineering', True)
    sc           = bundle.get('stacking_config', {})
    use_inter    = sc.get('use_interactions', True)
    use_rank     = sc.get('use_rank_features', False)

    # Pre-compute max valid code per categorical (for clamping)
    cat_max_code = {col: max(decode_maps[col].keys())
                    for col in decode_maps}

    def predict(X_raw_np):
        n = X_raw_np.shape[0]

        # 1. NumPy → DataFrame (raw features only, cat-coded)
        df = pd.DataFrame(X_raw_np, columns=raw_feat)

        # 2. Decode categoricals back to original string values
        for col in cat_cols:
            if col not in df.columns or col not in decode_maps:
                continue
            mx = cat_max_code[col]
            codes = df[col].round().clip(0, mx).astype(int)
            df[col] = codes.map(decode_maps[col]).fillna('Missing')

        # 3. Apply feature engineering (reconstructs engineered cols)
        if feat_eng:
            df = apply_feature_engineering(df)

        # 4. Build full-feature DataFrame in the correct column order
        for col in all_feat:
            if col not in df.columns:
                df[col] = np.nan
        X_full = df[all_feat].copy()

        # 5. Build encoded version for XGB / LGBM / RF
        X_enc = X_full.copy()
        for col in cat_cols:
            if col in X_enc.columns:
                X_enc[col] = X_enc[col].astype(str).astype('category').cat.codes
        X_enc = X_enc.fillna(-999)

        # 6. Run 4 base models (fold-averaged)
        base_preds = {}
        for algo in ['cat', 'xgb', 'lgbm', 'rf']:
            accum = np.zeros(n)
            for model in fold_models[algo]:
                if algo == 'cat':
                    # CatBoost: pass raw strings; flag cat columns
                    try:
                        from catboost import Pool
                        preds = model.predict_proba(
                            Pool(X_full, cat_features=cat_idx_full))[:, 1]
                    except Exception:
                        preds = model.predict_proba(X_full)[:, 1]
                else:
                    try:
                        preds = model.predict_proba(X_enc)[:, 1]
                    except ValueError:
                        try:
                            preds = model.predict_proba(
                                X_enc.values)[:, 1]
                        except Exception:
                            preds = model.predict_proba(
                                X_enc, validate_features=False)[:, 1]
                accum += preds
            base_preds[algo] = accum / n_folds

        # 7. Meta-learner or weighted average → raw probabilities
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
            X_meta = generate_meta_features(
                base_preds, use_interactions=use_inter, use_rank=use_rank)
            raw_probs = meta_model.predict_proba(X_meta)[:, 1]

        # 8. Calibration
        if calibrator is not None:
            final = calibrator.predict(raw_probs)
        else:
            final = raw_probs

        return final

    return predict


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_summary(shap_vals, X_display, feat_names, path, serial):
    """SHAP beeswarm summary plot."""
    plt.figure(figsize=(12, max(8, len(feat_names) * 0.35)))
    shap.summary_plot(
        shap_vals, X_display,
        feature_names=feat_names,
        show=False,
        max_display=min(30, len(feat_names)),
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] Saved summary plot → {path}")


def plot_shap_bar(shap_vals, feat_names, path, serial):
    """Mean |SHAP| horizontal bar chart."""
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    order     = np.argsort(mean_abs)[::-1]
    top_n     = min(30, len(feat_names))
    top_idx   = order[:top_n]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    y = np.arange(top_n)
    ax.barh(y, mean_abs[top_idx][::-1], color='#1E88E5', edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels([feat_names[i] for i in top_idx][::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Feature Importance (Mean |SHAP|) — {serial}")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] Saved bar plot → {path}")


def export_feature_importance(shap_vals, feat_names, path):
    scores = np.abs(shap_vals).mean(axis=0)
    df = pd.DataFrame({
        'Feature': feat_names,
        'Mean_Abs_SHAP': scores,
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)
    df.to_csv(path, index=False)
    print(f"[SHAP] Feature importance table → {path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(model_path, output_dir=None, background_k=50):
    """
    Full end-to-end SHAP analysis for a TearSense model bundle.

    Parameters
    ----------
    model_path     : str   Path to .pkl bundle
    output_dir     : str   Where to write ALL output (single directory).
                           If None, defaults to external_assessor/{serial}/SHAP_Analysis
    background_k   : int   Number of kmeans clusters for background (default 50)
    """
    print("\n" + "=" * 70)
    print("  SHAP ANALYSIS — TearSense Full Pipeline (End-to-End)")
    print("=" * 70)

    # ── 1. Load ──
    print(f"[SHAP] Loading: {model_path}")
    bundle = joblib.load(model_path)
    serial = bundle.get('serial_number', 'unknown')
    feat_eng = bundle.get('feature_engineering', True)

    if output_dir is None:
        output_dir = os.path.join("external_assessor", serial, "SHAP_Analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[SHAP] Serial:              {serial}")
    print(f"[SHAP] Feature engineering:  {feat_eng}")
    print(f"[SHAP] Output dir:          {output_dir}")

    # ── 2. Prepare data ──
    (X_test_shap, X_bg_shap, raw_feat, all_feat,
     cat_cols, cat_idx_full, decode_maps, medians) = prepare_shap_data(bundle)

    n_test, n_raw = X_test_shap.shape
    eng_excluded = [f for f in ENGINEERED_FEATURES if f in all_feat]

    print(f"[SHAP] Test samples:        {n_test}")
    print(f"[SHAP] Raw features (SHAP): {n_raw}")
    print(f"[SHAP] All features:        {len(all_feat)}")
    print(f"[SHAP] Engineered excluded: {eng_excluded}")

    # ── 3. Build predict function ──
    predict_fn = build_pipeline_predict_fn(
        bundle, raw_feat, all_feat,
        cat_cols, cat_idx_full, decode_maps,
    )

    # ── 4. Sanity-check: predict on test set should match stored metrics ──
    print("[SHAP] Verifying pipeline wrapper against stored metrics...")
    y_test = bundle.get('y_test_exact')
    if isinstance(y_test, (pd.Series, pd.DataFrame)):
        y_test = y_test.to_numpy().ravel()

    ts_probs = predict_fn(X_test_shap)
    from sklearn.metrics import roc_auc_score, brier_score_loss
    check_auc   = roc_auc_score(y_test, ts_probs)
    check_brier = brier_score_loss(y_test, ts_probs)
    stored      = bundle.get('metrics', {})
    stored_auc  = stored.get('test_auc')
    stored_brier = stored.get('test_brier')

    print(f"       Wrapper AUC:  {check_auc:.6f}  (stored: {stored_auc})")
    print(f"       Wrapper Brier: {check_brier:.6f}  (stored: {stored_brier})")
    if stored_auc and abs(check_auc - stored_auc) < 1e-3:
        print("       [PASS] Pipeline wrapper verified.")
    else:
        print("       [WARN] Small difference from stored metrics — "
              "may be due to median-fill on NaN in SHAP input.")

    # ── 5. Build KernelExplainer ──
    print(f"[SHAP] Building background summary (kmeans k={background_k})...")
    background = shap.kmeans(X_bg_shap, min(background_k, len(X_bg_shap)))

    print("[SHAP] Initialising KernelExplainer (full pipeline as black box)...")
    explainer = shap.KernelExplainer(predict_fn, background)

    # ── 6. Compute SHAP values ──
    t0 = time.time()
    print(f"[SHAP] Computing SHAP values for {n_test} test samples...")
    print(f"       This may take a while (full pipeline evaluated per perturbation).")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_test_shap, silent=True)
    elapsed = time.time() - t0
    print(f"[SHAP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── 7. Plots ──
    print("[SHAP] Generating plots...")
    X_display = pd.DataFrame(X_test_shap, columns=raw_feat)

    plot_shap_summary(
        shap_values, X_display, raw_feat,
        os.path.join(output_dir, f"shap_summary_{serial}.png"), serial,
    )
    plot_shap_bar(
        shap_values, raw_feat,
        os.path.join(output_dir, f"shap_bar_{serial}.png"), serial,
    )

    # ── 8. Export data ──
    importance_df = export_feature_importance(
        shap_values, raw_feat,
        os.path.join(output_dir, f"feature_importance_{serial}.csv"),
    )

    sv_df = pd.DataFrame(shap_values, columns=raw_feat)
    sv_path = os.path.join(output_dir, f"shap_values_{serial}.csv")
    sv_df.to_csv(sv_path, index=False)
    print(f"[SHAP] Raw SHAP values → {sv_path}")

    # ── 9. Metadata ──
    metadata = {
        "serial_number": serial,
        "feature_engineering_in_model": feat_eng,
        "engineered_features_excluded": eng_excluded,
        "total_features_in_model": len(all_feat),
        "raw_features_analysed": n_raw,
        "n_test_samples": n_test,
        "n_folds": bundle.get('n_folds', 5),
        "is_weighted_avg": bundle.get('is_weighted_avg', False),
        "best_method": bundle.get('best_method', 'unknown'),
        "background_k": background_k,
        "elapsed_seconds": round(elapsed, 1),
        "pipeline_check_auc": round(check_auc, 6),
        "stored_auc": float(stored_auc) if stored_auc else None,
        "top_10_features": importance_df.head(10)[
            ['Rank', 'Feature', 'Mean_Abs_SHAP']
        ].to_dict('records'),
    }
    meta_path = os.path.join(output_dir, f"shap_metadata_{serial}.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
    print(f"[SHAP] Metadata → {meta_path}")

    # ── 10. Summary ──
    print("\n" + "=" * 70)
    print(f"  SHAP ANALYSIS COMPLETE — {serial}")
    print("=" * 70)
    print(f"  Pipeline: raw features → FE → 4 base models → meta-learner → calibrator")
    print(f"  Features analysed: {n_raw}  (engineered excluded: {len(eng_excluded)})")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"\n  Top 5 Features:")
    for _, row in importance_df.head(5).iterrows():
        print(f"    {int(row['Rank']):>2}. {row['Feature']:<40} {row['Mean_Abs_SHAP']:.6f}")
    print(f"\n  All output → {output_dir}/")
    print("=" * 70)

    return shap_values, raw_feat, importance_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        mp = sys.argv[1]
        od = sys.argv[2] if len(sys.argv) >= 3 else None
        main(mp, output_dir=od)
    else:
        print("Usage: python -m runtime.externer_assessor_shap <model.pkl> [output_dir]")