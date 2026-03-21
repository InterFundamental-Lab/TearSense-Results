"""
P-value Table Generator — Intact vs Re-tear Group Comparisons
--------------------------------------------------------------
Produces the demographic/clinical comparison table:
  Variables          | Intact       | Retear       | P value
  ----------------------------------------------------------------
  Number of patients | n            | n            |
  Sex (no(%))        |              |              | chi-square
    Male             | n (%)        | n (%)        |
    Female           | n (%)        | n (%)        |
  Age                | mean ± SD    | mean ± SD    | Mann-Whitney / t-test
  Tear type (no(%))  |              |              | chi-square
    Full-Thickness   | n (%)        | n (%)        |
    Partial-Thickness| n (%)        | n (%)        |
  Tear-size area mm² | mean ± SD    | mean ± SD    | Mann-Whitney / t-test
  Insurance type     |              |              | chi-square
    Private          | n (%)        | n (%)        |
    Public           | n (%)        | n (%)        |
    Workers Comp     | n (%)        | n (%)        |
    Third Party      | n (%)        | n (%)        |
    DVA/Vet          | n (%)        | n (%)        |
    Self Funded      | n (%)        | n (%)        |

Statistical tests:
  - Continuous vars:  Shapiro-Wilk → if normal: t-test, else: Mann-Whitney U
  - Categorical vars: Chi-square (Fisher's exact if any expected count < 5)
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

TRAINING_CSV = os.path.join(
    os.path.dirname(__file__), "..", "TearSense-Trainer", "training.csv"
)

# ── Insurance type mapping (from valueMapping.txt) ──
INSURANCE_MAP = {
    0: "Private",
    1: "Workers Comp",
    2: "Third Party",
    3: "DVA/Vet",
    4: "Self Funded",
    5: "Public",
    6: "None/Unknown",
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    retear = pd.to_numeric(df["Re-Tear"], errors="coerce")
    df = df[retear.notna()].copy()
    df["Gender"] = df["Gender"].astype(str).str.strip().str.upper()
    df["Tear_Area"] = (
        pd.to_numeric(df["tear_characteristics_Tear_AntPost"], errors="coerce")
        * pd.to_numeric(df["tear_characteristics_Tear_MedLat"], errors="coerce")
    )
    df["Re-Tear"] = pd.to_numeric(df["Re-Tear"], errors="coerce").astype(int)
    df["InsuranceType"] = pd.to_numeric(df["InsuranceType"], errors="coerce")
    return df


def _fmt_n_pct(count, total):
    pct = count / total * 100 if total > 0 else 0
    return f"{count} ({pct:.1f}%)"


def _fmt_mean_sd(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "N/A"
    return f"{s.mean():.1f} \u00b1 {s.std():.1f}"


def _test_continuous(intact_vals, retear_vals):
    """Pick t-test or Mann-Whitney based on normality (Shapiro-Wilk)."""
    a = intact_vals.dropna()
    b = retear_vals.dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan, "N/A"
    # Shapiro-Wilk on a subsample (limit to 5000 for performance)
    _, p_norm_a = stats.shapiro(a.sample(min(len(a), 5000), random_state=42))
    _, p_norm_b = stats.shapiro(b.sample(min(len(b), 5000), random_state=42))
    if p_norm_a > 0.05 and p_norm_b > 0.05:
        stat, p = stats.ttest_ind(a, b, equal_var=False)
        return p, "t-test"
    else:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return p, "Mann-Whitney U"


def _test_categorical(intact_counts, retear_counts):
    """Chi-square or Fisher's exact test for a contingency table."""
    table = np.array([intact_counts, retear_counts])
    # Remove columns where both groups are 0
    mask = table.sum(axis=0) > 0
    table = table[:, mask]
    if table.shape[1] < 2:
        return np.nan, "N/A"
    expected = stats.contingency.expected_freq(table)
    if (expected < 5).any():
        # Fisher's exact for 2×2, chi-square with simulated p for larger
        if table.shape == (2, 2):
            _, p = stats.fisher_exact(table)
            return p, "Fisher's exact"
    stat, p, dof, _ = stats.chi2_contingency(table)
    return p, "Chi-square"


def _fmt_p(p, test_name=""):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "N/A"
    if p < 0.001:
        return f"<0.001 ({test_name})" if test_name else "<0.001"
    return f"{p:.3f} ({test_name})" if test_name else f"{p:.3f}"


def generate_table(df: pd.DataFrame):
    intact = df[df["Re-Tear"] == 0]
    retear = df[df["Re-Tear"] == 1]
    n_intact = len(intact)
    n_retear = len(retear)

    rows = []
    W = 25  # label width

    # ── Number of patients ──
    rows.append(("Number of patients", str(n_intact), str(n_retear), ""))

    # ── Sex ──
    m_intact = (intact["Gender"] == "M").sum()
    f_intact = (intact["Gender"] == "F").sum()
    m_retear = (retear["Gender"] == "M").sum()
    f_retear = (retear["Gender"] == "F").sum()
    p_sex, test_sex = _test_categorical(
        [m_intact, f_intact], [m_retear, f_retear]
    )
    rows.append(("Sex (no(%))", "", "", _fmt_p(p_sex, test_sex)))
    rows.append(("  Male", _fmt_n_pct(m_intact, n_intact), _fmt_n_pct(m_retear, n_retear), ""))
    rows.append(("  Female", _fmt_n_pct(f_intact, n_intact), _fmt_n_pct(f_retear, n_retear), ""))

    # ── Age ──
    age_intact = pd.to_numeric(intact["Aged"], errors="coerce")
    age_retear = pd.to_numeric(retear["Aged"], errors="coerce")
    p_age, test_age = _test_continuous(age_intact, age_retear)
    rows.append(("Age", _fmt_mean_sd(age_intact), _fmt_mean_sd(age_retear), _fmt_p(p_age, test_age)))

    # ── Tear type ──
    full_intact = (intact["tear_characteristics_Full"] == 3).sum()
    part_intact = (intact["tear_characteristics_Full"] == 2).sum()
    full_retear = (retear["tear_characteristics_Full"] == 3).sum()
    part_retear = (retear["tear_characteristics_Full"] == 2).sum()
    p_tear, test_tear = _test_categorical(
        [full_intact, part_intact], [full_retear, part_retear]
    )
    rows.append(("Tear type (no(%))", "", "", _fmt_p(p_tear, test_tear)))
    rows.append(("  Full-Thickness", _fmt_n_pct(full_intact, n_intact), _fmt_n_pct(full_retear, n_retear), ""))
    rows.append(("  Partial-Thickness", _fmt_n_pct(part_intact, n_intact), _fmt_n_pct(part_retear, n_retear), ""))

    # ── Tear-size area ──
    area_intact = intact["Tear_Area"]
    area_retear = retear["Tear_Area"]
    p_area, test_area = _test_continuous(area_intact, area_retear)
    rows.append((
        "Tear-size area (mm\u00b2)",
        _fmt_mean_sd(area_intact),
        _fmt_mean_sd(area_retear),
        _fmt_p(p_area, test_area),
    ))

    # ── Insurance type ──
    intact_ins_counts = []
    retear_ins_counts = []
    ins_labels = []
    for code, label in sorted(INSURANCE_MAP.items()):
        ins_labels.append(label)
        intact_ins_counts.append((intact["InsuranceType"] == code).sum())
        retear_ins_counts.append((retear["InsuranceType"] == code).sum())

    p_ins, test_ins = _test_categorical(intact_ins_counts, retear_ins_counts)
    rows.append(("Insurance type", "", "", _fmt_p(p_ins, test_ins)))
    for label, ic, rc in zip(ins_labels, intact_ins_counts, retear_ins_counts):
        rows.append((f"  {label}", _fmt_n_pct(ic, n_intact), _fmt_n_pct(rc, n_retear), ""))

    return rows


def print_table(rows):
    c1, c2, c3, c4 = 22, 20, 20, 28
    sep = "-" * (c1 + c2 + c3 + c4 + 7)
    header = f"{'Variables':<{c1}} | {'Intact':<{c2}} | {'Retear':<{c3}} | {'P value':<{c4}}"
    print(sep)
    print(header)
    print(sep)
    for label, col_intact, col_retear, p_val in rows:
        print(f"{label:<{c1}} | {col_intact:<{c2}} | {col_retear:<{c3}} | {p_val:<{c4}}")
    print(sep)


def export_csv(rows, path):
    df_out = pd.DataFrame(rows, columns=["Variables", "Intact", "Retear", "P value"])
    df_out.to_csv(path, index=False)
    print(f"\nTable exported to: {path}")


def main():
    df = load_data(TRAINING_CSV)
    rows = generate_table(df)
    print_table(rows)

    out_path = os.path.join(os.path.dirname(__file__), "outputs", "p_value_table.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_csv(rows, out_path)


if __name__ == "__main__":
    main()
