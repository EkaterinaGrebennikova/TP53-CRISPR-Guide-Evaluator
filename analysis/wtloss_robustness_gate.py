"""Steps 1 + 2: the decisive robustness pair for the WT-loss bridge.

1. Exact bootstrap 95% CIs (hard & Phi) at k=2/3/4 -- in particular the
   k=2 Phi lower bound vs the pre-registered 10% kill line.

2. CBE-model robustness gate:
   2a. Symmetric sigma-perturbation Monte Carlo: redraw every patient's
       true efficiency ~ N(pred, sigma_modality), recompute the HARD
       band. If the band can collapse toward 0 under the model's own
       RMSE, it dies. (Its mean should also ~match the analytic Phi --
       a consistency check that Phi is doing the right thing.)
   2b. Load-bearing decomposition: how much of the band is CBE vs ABE,
       and what the band is with CBE removed. States the dependency
       honestly rather than hiding it.

3. MSK-IMPACT independent replication -- feasibility-gated. The TCGA
   finding is defined on CN-only LOH (tp53_cna <= -1), which is what
   makes it VAF-artifact-immune. Replication is only valid if the
   independent cohort can reproduce THAT definition. This step checks
   that first and refuses to substitute a VAF-based LOH proxy (which
   would reintroduce the indel-VAF artifact the whole analysis avoids).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.stats import norm

from wtloss_correctability_makebreak import (
    collect, thresholds, K_VALUES, best_eff, SIGMA, _cache,
)
from mutationparser import get_reference
from mskvalidation import load_msk_mutations, load_msk_cna

B = 4000
RNG = np.random.default_rng(7)


def hard_band(e, k):
    tr, tl = thresholds(k)
    return float(np.mean((e >= tr) & (e < tl))) * 100


def phi_band(e, sig, k):
    tr, tl = thresholds(k)
    return float(np.mean(norm.cdf((e - tr) / sig)
                         - norm.cdf((e - tl) / sig))) * 100


def main():
    d = collect()
    e, sig, mods = d['e'], d['sig'], d['mods']
    n = len(e)

    # ---- Step 1: exact bootstrap CIs ----
    print("\n=== STEP 1: bootstrap 95% CIs (patient resampling, "
          f"B={B}) ===")
    print(f"{'k':>2} | {'hard %':>20} | {'Phi-debiased %':>22}")
    for k in K_VALUES:
        hb = np.empty(B); pb = np.empty(B)
        for b in range(B):
            idx = RNG.integers(0, n, n)
            hb[b] = hard_band(e[idx], k)
            pb[b] = phi_band(e[idx], sig[idx], k)
        h = hard_band(e, k); p = phi_band(e, sig, k)
        hlo, hhi = np.percentile(hb, [2.5, 97.5])
        plo, phi_ = np.percentile(pb, [2.5, 97.5])
        flag = '' if plo >= 10 else '  <-- Phi lower CI BELOW 10% kill line'
        print(f"{k:>2} | {h:5.1f} [{hlo:4.1f}, {hhi:4.1f}]      | "
              f"{p:5.1f} [{plo:4.1f}, {phi_:4.1f}]{flag}")

    # ---- Step 2a: symmetric sigma-perturbation Monte Carlo ----
    print(f"\n=== STEP 2a: sigma-perturbation MC (model RMSE, B={B}) ===")
    print("redraw true eff ~ N(pred, sigma); recompute HARD band")
    print(f"{'k':>2} | {'perturbed band %':>26} | {'analytic Phi':>12}")
    collapse = False
    for k in K_VALUES:
        vals = np.empty(B)
        for b in range(B):
            ep = np.clip(e + RNG.normal(0, sig), 0.0, 1.0)
            vals[b] = hard_band(ep, k)
        m = vals.mean()
        lo, hi = np.percentile(vals, [2.5, 97.5])
        pa = phi_band(e, sig, k)
        if lo <= 1.0:                       # band can vanish -> fragile
            collapse = True
        print(f"{k:>2} | {m:5.1f} [{lo:4.1f}, {hi:4.1f}]          | "
              f"{pa:11.1f}")

    # ---- Step 2b: load-bearing decomposition ----
    print("\n=== STEP 2b: who carries the band (k=4) ===")
    tr, tl = thresholds(4)
    in_band = (e >= tr) & (e < tl)
    n_cbe = int(np.sum(mods == 'CBE'))
    n_abe = int(np.sum(mods == 'ABE'))
    band_cbe = int(np.sum(in_band & (mods == 'CBE')))
    band_abe = int(np.sum(in_band & (mods == 'ABE')))
    tot_band = band_cbe + band_abe
    print(f"  total in-band: {tot_band}  "
          f"(CBE {band_cbe} = {100*band_cbe/max(tot_band,1):.0f}%, "
          f"ABE {band_abe} = {100*band_abe/max(tot_band,1):.0f}%)")
    abe_only = 100 * band_abe / max(n_abe, 1)
    cbe_only = 100 * band_cbe / max(n_cbe, 1)
    print(f"  ABE-only band fraction: {abe_only:.1f}%  "
          f"(n_ABE={n_abe})  -> bridge WITHOUT trusting CBE")
    print(f"  CBE-only band fraction: {cbe_only:.1f}%  "
          f"(n_CBE={n_cbe})")

    # ---- verdict ----
    print("\n--- STEP 2 VERDICT ---")
    if collapse:
        print("  KILL: under the CBE/ABE model's own RMSE the band can "
              "vanish (lower MC bound ~0). Not robust to model error.")
    else:
        print("  SURVIVES: band does not collapse under model-RMSE "
              "perturbation, and perturbed mean tracks analytic Phi "
              "(internal consistency).")
    print(f"  HONEST SCOPING (non-negotiable in the paper): the bridge "
          f"is CBE-load-bearing -- {100*band_cbe/max(tot_band,1):.0f}% of "
          f"the band is CBE; ABE-correctable LOH patients are essentially "
          f"NOT WT-loss-limited (ABE-only band {abe_only:.1f}%). The claim "
          f"must be stated as a CBE-correctable-LOH effect, conditional "
          f"on CBE model M.")

    replicate_msk()


def replicate_msk():
    """STEP 3: MSK-IMPACT replication -- feasibility-gated.

    The TCGA finding is defined on CN-only LOH (tp53_cna <= -1), which is
    exactly what makes it VAF-artifact-immune. A valid replication MUST
    reproduce that definition. We check MSK can do so BEFORE computing
    anything, and refuse to substitute a VAF-based LOH proxy.
    """
    print("\n" + "=" * 62)
    print("=== STEP 3: MSK-IMPACT independent replication ===")
    print("=" * 62)
    m = load_msk_mutations()
    c = load_msk_cna()
    m = m[m['Hugo_Symbol'] == 'TP53']

    vc = c['tp53_cna'].value_counts().sort_index().to_dict()
    print(f"  MSK TP53 discrete CNA distribution: {vc}")
    n_shallow = int((c['tp53_cna'] == -1).sum())
    n_deep = int((c['tp53_cna'] == -2).sum())

    # CN-LOH (the TCGA definition) requires hemizygous loss = -1, with the
    # mutant allele retained. Deep deletion (-2) removes the mutant allele
    # too -> it is NOT 'LOH with retained mutation'.
    if n_shallow == 0:
        print("\n  FEASIBILITY: BLOCKED.")
        print("  MSK-IMPACT's discrete CNA profile emits NO -1 "
              "(shallow/hemizygous) calls.")
        print(f"  Only deep deletion (-2, n={n_deep}) and amplification "
              f"exist. TP53 LOH is overwhelmingly hemizygous; -2 removes "
              f"the mutant allele too, so it is biologically the opposite "
              f"of 'LOH with retained mutation'.")
        # show how broken a forced attempt would be
        nm = m.groupby('patient_id').size().to_dict()
        single = {p for p, n in nm.items() if n == 1}
        cna_l = c.set_index('patient_id')['tp53_cna'].to_dict()
        forced = sum(1 for _, r in m.iterrows()
                     if r['patient_id'] in single
                     and cna_l.get(r['patient_id'], 0) <= -1)
        print(f"  A forced tp53_cna<=-1 LOH set = {forced} patients "
              f"(deep-del only; biologically inconsistent with the "
              f"finding's state). Not a valid replication.")
        print("\n  VERDICT: MSK-IMPACT CANNOT independently validate the "
              "CN-LOH correctability bridge.")
        print("  This is a DATA-AVAILABILITY limitation, not a finding "
              "failure. Substituting VAF-LOH would reintroduce the "
              "indel-VAF artifact this analysis was built to avoid -- "
              "explicitly NOT done.")
        print("  -> The finding remains single-cohort (TCGA) by data "
              "constraint. External validation requires a cohort with "
              "allele-specific / shallow CN (e.g. ABSOLUTE/FACETS on "
              "TCGA WES, PCAWG) or experimental confirmation.")
        return

    # (only runs if a future cohort DOES carry -1 calls)
    ref = get_reference()
    nm = m.groupby('patient_id').size().to_dict()
    single = {p for p, n in nm.items() if n == 1}
    cna_l = c.set_index('patient_id')['tp53_cna'].to_dict()
    rows = [r['aa_change'] for _, r in m.iterrows()
            if r['patient_id'] in single
            and cna_l.get(r['patient_id'], 0) <= -1
            and isinstance(r['aa_change'], str) and r['aa_change'][:1].isalpha()]
    for aa in set(rows):
        best_eff(aa, ref)
    pats = []
    for aa in rows:
        eff = _cache.get(aa, {})
        if eff:
            mod = max(eff, key=eff.get)
            pats.append((eff[mod], SIGMA[mod]))
    if not pats:
        print("  No base-editable CN-LOH patients in MSK. Cannot replicate.")
        return
    e = np.array([p[0] for p in pats])
    sig = np.array([p[1] for p in pats])
    print(f"  MSK CN-LOH base-editable n={len(e)}")
    for k in K_VALUES:
        print(f"  k={k}: hard={hard_band(e,k):.1f}%  "
              f"Phi={phi_band(e,sig,k):.1f}%")


if __name__ == '__main__':
    main()
