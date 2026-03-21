"""
Cohort Demographics & Intraoperative Findings Analysis
-------------------------------------------------------
Generates the descriptive statistics paragraph for the manuscript:
  - Gender distribution
  - Mean age ± SD
  - Partial vs full thickness tear counts and mean area ± SD
  - Mean number of anchors ± SD
  - Re-tear cases at 6 months postoperatively
"""

import os
import pandas as pd
import numpy as np

TRAINING_CSV = os.path.join(
    os.path.dirname(__file__), "..", "TearSense-Trainer", "training.csv"
)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Match core.py filtering: drop rows where Re-Tear is NaN
    # (these patients are excluded from training)
    retear = pd.to_numeric(df["Re-Tear"], errors="coerce")
    df = df[retear.notna()].copy()
    # Normalise gender values (strip whitespace, uppercase)
    df["Gender"] = df["Gender"].astype(str).str.strip().str.upper()
    # Compute tear area (mm²) = AntPost × MedLat
    df["Tear_Area"] = (
        pd.to_numeric(df["tear_characteristics_Tear_AntPost"], errors="coerce")
        * pd.to_numeric(df["tear_characteristics_Tear_MedLat"], errors="coerce")
    )
    return df


def cohort_summary(df: pd.DataFrame) -> dict:
    """Return a dict with all descriptive statistics."""
    stats = {}

    # --- Gender ---
    stats["n_total"] = len(df)
    stats["n_female"] = (df["Gender"] == "F").sum()
    stats["n_male"] = (df["Gender"] == "M").sum()

    # --- Age ---
    age = pd.to_numeric(df["Aged"], errors="coerce").dropna()
    stats["age_mean"] = age.mean()
    stats["age_std"] = age.std()

    # --- Tear type (2 = partial, 3 = full) ---
    partial = df[df["tear_characteristics_Full"] == 2]
    full = df[df["tear_characteristics_Full"] == 3]

    stats["n_partial"] = len(partial)
    partial_area = partial["Tear_Area"].dropna()
    stats["partial_area_mean"] = partial_area.mean()
    stats["partial_area_std"] = partial_area.std()

    stats["n_full"] = len(full)
    full_area = full["Tear_Area"].dropna()
    stats["full_area_mean"] = full_area.mean()
    stats["full_area_std"] = full_area.std()

    # --- Anchors ---
    anchors = pd.to_numeric(df["No_of_anchors"], errors="coerce").dropna()
    stats["anchors_mean"] = anchors.mean()
    stats["anchors_std"] = anchors.std()

    # --- Re-tear at 6 months ---
    retear = pd.to_numeric(df["Re-Tear"], errors="coerce")
    stats["n_retear"] = int(retear.sum())
    stats["n_retear_evaluated"] = int(retear.notna().sum())

    return stats


def format_paragraph(s: dict) -> str:
    return (
        f"The cohort was composed of {s['n_female']} females and {s['n_male']} males "
        f"and the mean age of the cohort was {s['age_mean']:.1f} \u00b1 {s['age_std']:.1f} years. "
        f"Intraoperative findings identified {s['n_partial']} partial thickness tear cases "
        f"(mean area: {s['partial_area_mean']:.1f} \u00b1 {s['partial_area_std']:.1f} mm\u00b2) "
        f"and {s['n_full']} full thickness tear cases "
        f"(mean area: {s['full_area_mean']:.1f} \u00b1 {s['full_area_std']:.1f} mm\u00b2). "
        f"The mean number of anchors was {s['anchors_mean']:.1f} \u00b1 {s['anchors_std']:.1f}. "
        f"At 6 months postoperatively, structural healing was evaluated via ultrasound, "
        f"and {s['n_retear']} re-tear cases were recorded "
        f"(out of {s['n_retear_evaluated']} patients evaluated)."
    )


def main():
    df = load_data(TRAINING_CSV)
    stats = cohort_summary(df)

    print("=" * 70)
    print("COHORT DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(f"  Total patients:          {stats['n_total']}")
    print(f"  Females:                 {stats['n_female']}")
    print(f"  Males:                   {stats['n_male']}")
    print(f"  Age (mean \u00b1 SD):         {stats['age_mean']:.1f} \u00b1 {stats['age_std']:.1f} years")
    print()
    print(f"  Partial thickness tears: {stats['n_partial']}")
    print(f"    Area (mean \u00b1 SD):      {stats['partial_area_mean']:.1f} \u00b1 {stats['partial_area_std']:.1f} mm\u00b2")
    print(f"  Full thickness tears:    {stats['n_full']}")
    print(f"    Area (mean \u00b1 SD):      {stats['full_area_mean']:.1f} \u00b1 {stats['full_area_std']:.1f} mm\u00b2")
    print()
    print(f"  Anchors (mean \u00b1 SD):     {stats['anchors_mean']:.1f} \u00b1 {stats['anchors_std']:.1f}")
    print(f"  Re-tear cases (6mo):     {stats['n_retear']} / {stats['n_retear_evaluated']}")
    print()
    print("=" * 70)
    print("MANUSCRIPT PARAGRAPH")
    print("=" * 70)
    print(format_paragraph(stats))
    print()


if __name__ == "__main__":
    main()
