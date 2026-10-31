"""
Item 3: the paper reports 5 formal significance tests (Sections 5.1, 5.2,
5.3, 5.6a, 5.6b) with no correction for multiple comparisons. Individually
each p-value is a legitimate answer to its own question, but treating a
family of 5 tests each at alpha=0.05 means an appreciable chance that at
least one looks "significant" by chance even if every null hypothesis were
true. This applies Holm-Bonferroni correction (stronger than naive
Bonferroni; controls family-wise error rate without being overly
conservative) across all 5, and reports which survive.

Run this after multiseed_stability.py and feature_access_and_model_family_check.py
so multiseed_paired_comparisons.csv, multiseed_ablation.csv,
mlp_model_family_ablation.csv, and feature_access_ablation.csv all exist.
"""
import numpy as np
import pandas as pd
from scipy import stats as sstats
from statsmodels.stats.multitest import multipletests

from config import OUTPUT_DIR


def main():
    paired = pd.read_csv(OUTPUT_DIR / "multiseed_paired_comparisons.csv")
    ablation_53 = pd.read_csv(OUTPUT_DIR / "multiseed_ablation.csv")
    ablation_56a = pd.read_csv(OUTPUT_DIR / "mlp_model_family_ablation.csv")
    ablation_56b = pd.read_csv(OUTPUT_DIR / "feature_access_ablation.csv")

    tests = []

    t, p = sstats.ttest_rel(paired["rf_f1"], paired["xgb_f1"])
    tests.append(("5.1 Random Forest vs XGBoost F1", t, p, len(paired)))

    t, p = sstats.ttest_rel(paired["cv_selected_ml_f1"], paired["hrrn_f1"])
    tests.append(("5.2 CV-selected ML vs HRRN F1", t, p, len(paired)))

    t, p = sstats.ttest_1samp(ablation_53["delta"], 0.0)
    tests.append(("5.3 pk_wait_anomaly ablation (RF, full access)", t, p, len(ablation_53)))

    t, p = sstats.ttest_1samp(ablation_56a["delta_full_vs_raw"], 0.0)
    tests.append(("5.6(a) pk_wait_anomaly x MLP (full access)", t, p, len(ablation_56a)))

    t, p = sstats.ttest_1samp(ablation_56b["delta"], 0.0)
    tests.append(("5.6(b) pk_wait_anomaly, reduced feature access", t, p, len(ablation_56b)))

    names = [x[0] for x in tests]
    pvals = [x[2] for x in tests]

    reject_holm, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    reject_bonf, p_bonf, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")
    reject_bh, p_bh, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    summary = pd.DataFrame({
        "test": names,
        "n_seeds": [x[3] for x in tests],
        "t_stat": [round(x[1], 3) for x in tests],
        "p_raw": [round(p, 5) for p in pvals],
        "sig_raw_0.05": [p < 0.05 for p in pvals],
        "p_holm": [round(p, 5) for p in p_holm],
        "sig_holm_0.05": reject_holm,
        "p_bonferroni": [round(p, 5) for p in p_bonf],
        "sig_bonferroni_0.05": reject_bonf,
        "p_benjamini_hochberg": [round(p, 5) for p in p_bh],
        "sig_bh_0.05": reject_bh,
    })
    summary.to_csv(str(OUTPUT_DIR / "multiple_comparisons_correction.csv"), index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 45)
    print(summary.to_string(index=False))
    print(
        "\nHolm-Bonferroni (family of 5, alpha=0.05) is the primary correction reported in "
        "the paper -- it controls the family-wise error rate without Bonferroni's full "
        "conservatism. Benjamini-Hochberg (controls false discovery rate, more permissive) "
        "and plain Bonferroni are included for reference."
    )


if __name__ == "__main__":
    main()
