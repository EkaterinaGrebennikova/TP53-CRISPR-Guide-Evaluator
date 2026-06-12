"""MAKE-OR-BREAK TEST (pre-registered) for the WT-loss correctability bridge.

Metric: WT-loss-attributable non-correctability.
Among REAL CN-defined LOH single-mutation patients with a base-editable
mutation, the fraction whose best-guide ML efficiency e falls in
  [ t_ret(k) , t_lost(k) )
i.e. their own guide WOULD restore function in a WT-retained tumor but
does NOT once the WT allele is lost.

  t_ret(k)  = 2 * 0.45**(1/k) - 1     # (0.5 + 0.5 e)^k >= 0.45
  t_lost(k) = 0.45**(1/k)             #          e^k    >= 0.45

Allelic state: GISTIC CN ONLY (tp53_cna <= -1 = WT-lost). No VAF gate.
Single-mutation patients only. Best-by-efficiency selection (most
generous -> conservative for a kill test). Phi-debiasing with per-
modality held-out residual SD applied symmetrically.

PRE-REGISTERED KILL CRITERION:
  If the WT-loss-attributable fraction among LOH base-editable patients
  falls below 10% at the standard exponent (k=4) under EITHER the hard
  count OR the Phi-debiased estimate (the figure-reported kill line),
  the bridge is dead. No build.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.stats import norm

from tcgaloader import load_mutations, load_cna
from mutationparser import parse_mutations, get_reference
from modalityselector import select_modalities
from guidedesigner import design_guide
from guidescorer import score_guide

SIGMA = {'ABE': 0.2013, 'CBE': 0.1932}   # group-CV RMSE of deployed models
                                          # (see analysis/extract_honest_sigma.py)
                                          # previously 0.156 / 0.151 from leaky
                                          # random-split residuals
K_VALUES = [2, 3, 4]


def thresholds(k):
    root = 0.45 ** (1.0 / k)
    return max(0.0, 2.0 * root - 1.0), root        # t_ret, t_lost


_cache = {}

def best_eff(aa, ref):
    if aa in _cache:
        return _cache[aa]
    out = {}
    try:
        parsed = parse_mutations([aa])
    except Exception:
        parsed = []
    if parsed:
        m = parsed[0]
        for it in select_modalities(m):
            mod, nt = it['modality'], it['nt_change']
            if mod not in ('ABE', 'CBE'):
                continue
            try:
                guides = design_guide(nt, mod, ref.cds_sequence)
            except Exception:
                continue
            effs = []
            for g in guides:
                try:
                    score_guide(g, mod, nt, ref.cds_sequence)
                except Exception:
                    continue
                e = g.get('ml_efficiency')
                if e is not None:
                    effs.append(e)
            if effs:
                out[mod] = max(out.get(mod, 0.0), max(effs))
    _cache[aa] = out
    return out


def collect(verbose=True):
    """Run the pipeline once; return per-patient arrays for real CN-LOH
    single-mutation base-editable patients.
      e    : best-guide ML efficiency (clinically-optimal base editor)
      sig  : per-patient model residual SD (by chosen modality)
      mods : chosen modality per patient ('ABE'/'CBE')
    plus cohort counts.
    """
    muts = load_mutations()
    cna = load_cna()
    ref = get_reference()
    n_muts = muts.groupby('patient_id').size().to_dict()
    single = {p for p, n in n_muts.items() if n == 1}
    cna_l = cna.set_index('patient_id')['tp53_cna'].to_dict()

    loh_rows = []
    n_loh_total = 0
    for _, r in muts.iterrows():
        pid = r['patient_id']
        if pid not in single:
            continue
        cn = cna_l.get(pid, 0)
        if cn > -1:                       # CN-only WT-loss definition
            continue
        n_loh_total += 1
        aa = r.get('aa_change', '')
        if isinstance(aa, str) and aa and aa[0].isalpha():
            loh_rows.append(aa)

    uniq = sorted(set(loh_rows))
    if verbose:
        print(f"CN-defined LOH single-mutation patients: {n_loh_total}")
        print(f"  with parseable AA change: {len(loh_rows)}  "
              f"(unique {len(uniq)})")
    for i, aa in enumerate(uniq):
        best_eff(aa, ref)
        if verbose and (i + 1) % 100 == 0:
            print(f"  designed {i+1}/{len(uniq)}")

    pats = []
    n_pe_hdr = 0
    for aa in loh_rows:
        eff = _cache.get(aa, {})
        if not eff:
            n_pe_hdr += 1               # PE/HDR-only: not base-editable
            continue
        mod = max(eff, key=eff.get)
        pats.append((eff[mod], mod))
    e = np.array([p[0] for p in pats])
    mods = np.array([p[1] for p in pats])
    sig = np.array([SIGMA[p[1]] for p in pats])
    if verbose:
        print(f"  base-editable (ABE/CBE): {len(e)}   "
              f"PE/HDR-only (no base editor): {n_pe_hdr}")
    return {'e': e, 'sig': sig, 'mods': mods,
            'n_loh_total': n_loh_total, 'n_be': len(e),
            'n_pe_hdr': n_pe_hdr}


def main():
    d = collect()
    e, sig = d['e'], d['sig']

    print("\n=== WT-loss-attributable non-correctability "
          "(LOH base-editable patients) ===")
    print(f"{'k':>2} {'t_ret':>6} {'t_lost':>6} "
          f"{'clears LOH':>11} {'in band':>9} {'below t_ret':>12} "
          f"{'band Phi-debiased':>18}")
    verdict = {}
    for k in K_VALUES:
        tr, tl = thresholds(k)
        clears = float(np.mean(e >= tl)) * 100
        in_band = float(np.mean((e >= tr) & (e < tl))) * 100
        below = float(np.mean(e < tr)) * 100
        # Phi-debiased: E[ P(t_ret <= true_e < t_lost | pred e, sigma) ]
        phi_band = float(np.mean(norm.cdf((e - tr) / sig)
                                 - norm.cdf((e - tl) / sig))) * 100
        verdict[k] = (in_band, phi_band)
        print(f"{k:>2} {tr:6.3f} {tl:6.3f} {clears:10.1f}% "
              f"{in_band:8.1f}% {below:11.1f}% {phi_band:17.1f}%")

    ib4, phi4 = verdict[4]
    print("\n--- PRE-REGISTERED VERDICT (standard exponent k=4) ---")
    print(f"  WT-loss-attributable fraction (hard):        {ib4:.1f}%")
    print(f"  WT-loss-attributable fraction (Phi-debiased): {phi4:.1f}%")
    kill = (ib4 < 10.0) or (phi4 < 10.0)
    if kill:
        print("  VERDICT: KILL. Band unpopulated / collapses under "
              "uncertainty. Allelic state is decision-irrelevant given "
              "modality+efficiency. Do NOT build the bridge.")
    else:
        print("  VERDICT: PASSES make-or-break. Band is populated and "
              "survives Phi-debiasing -> WT-loss materially changes the "
              "CRISPR correctability verdict. Proceed to full scope.")


if __name__ == '__main__':
    main()
