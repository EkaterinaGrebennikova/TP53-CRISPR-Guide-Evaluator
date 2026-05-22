"""2D rescuability landscape: modality x allelic state.

For each TCGA TP53-mutant patient:
  - allelic state (single-mutation: het CN-neutral / het+gain / LOH;
    multi-mutation handled separately for the biallelic argument)
  - required correction modality (ABE / CBE / PE / HDR)
  - best deployable guide's ML-predicted AA-correction efficiency

Rescuability rule (composed allele + tetramer model):
  effective_wt(het)  = 0.5 + 0.5*eff      (one WT allele retained)
  effective_wt(LOH)  = eff                (no WT allele)
  tetramer_fraction  = effective_wt ** k
  rescuable iff tetramer_fraction >= 0.45  (Ventura)

Robustness band: k in {2, 3, 4} (tetramer dominant-negative strength).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from tcgaloader import load_mutations, load_cna
from tcgaallelic import compute_vaf, classify_allelic_state, load_purity, PURITY_FILE
from mutationparser import parse_mutations, get_reference
from modalityselector import select_modalities
from guidedesigner import design_guide
from guidescorer import score_guide

K_VALUES = [2, 3, 4]
TET_THRESHOLD = 0.45


def thresholds_for_k(k):
    """Min ML efficiency to reach 0.45 tetramer, per allelic state."""
    root = TET_THRESHOLD ** (1.0 / k)
    return {
        'het': max(0.0, 2.0 * root - 1.0),   # (0.5 + 0.5*eff)^k >= 0.45
        'loh': root,                          # eff^k >= 0.45
    }


# ---- best deployable guide per (aa_change, modality) ----
_eff_cache = {}

def best_guide_efficiency(aa_change, ref):
    """Return {modality: max ml_efficiency} over designed guides.
    Modalities limited to ABE/CBE/PE (HDR has no efficiency model)."""
    if aa_change in _eff_cache:
        return _eff_cache[aa_change]
    out = {}
    try:
        parsed = parse_mutations([aa_change])
    except Exception:
        parsed = []
    if not parsed:
        _eff_cache[aa_change] = out
        return out
    m = parsed[0]
    for item in select_modalities(m):
        mod, nt = item['modality'], item['nt_change']
        if mod not in ('ABE', 'CBE', 'Prime Editing'):
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
    _eff_cache[aa_change] = out
    return out


def aa_from_row(row):
    aa = row.get('aa_change', '')
    return aa if isinstance(aa, str) and aa and aa[0].isalpha() else None


def main():
    print("Loading TCGA + reference...")
    muts = load_mutations()
    cna = load_cna()
    ref = get_reference()
    n_muts = muts.groupby('patient_id').size().to_dict()
    cna_l = cna.set_index('patient_id')['tp53_cna'].to_dict()
    pur = load_purity() if os.path.exists(PURITY_FILE) else {}

    single_pids = {p for p, n in n_muts.items() if n == 1}

    state_map = {
        'loh_with_mutation': 'loh',
        'heterozygous_cn_neutral': 'het',
        'heterozygous_with_gain': 'het',   # retains a WT allele (conservative)
    }
    grid = {}            # (modality, state) -> list of best-eff per patient
    uniq = set()
    for _, row in muts.iterrows():
        pid = row['patient_id']
        if pid not in single_pids:
            continue
        aa = aa_from_row(row)
        if aa is None:
            continue
        vaf = compute_vaf(row, purity=pur.get(pid))
        raw_state = classify_allelic_state(vaf, cna_l.get(pid, 0), 1)
        if state_map.get(raw_state):
            uniq.add(aa)

    print(f"Unique single-mutation missense AA changes to design: {len(uniq)}")
    for i, aa in enumerate(sorted(uniq)):
        best_guide_efficiency(aa, ref)
        if (i + 1) % 100 == 0:
            print(f"  designed {i+1}/{len(uniq)}")

    for _, row in muts.iterrows():
        pid = row['patient_id']
        if pid not in single_pids:
            continue
        aa = aa_from_row(row)
        if aa is None:
            continue
        vaf = compute_vaf(row, purity=pur.get(pid))
        raw_state = classify_allelic_state(vaf, cna_l.get(pid, 0), 1)
        st = state_map.get(raw_state)
        if st is None:
            continue
        effs = _eff_cache.get(aa, {})
        if not effs:
            continue
        mod = max(effs, key=effs.get)   # clinically optimal modality
        grid.setdefault((mod, st), []).append(effs[mod])

    mods = ['ABE', 'CBE', 'Prime Editing']
    states = ['het', 'loh']
    print("\n" + "=" * 74)
    print("2D RESCUABILITY LANDSCAPE  (single-mutation patients)")
    print("cell = % of patients whose best guide clears 0.45 tetramer "
          "[k=2 / k=3 / k=4]")
    print("=" * 74)
    thr = {k: thresholds_for_k(k) for k in K_VALUES}
    for mod in mods:
        for st in states:
            v = np.array(grid.get((mod, st), []))
            if len(v) == 0:
                print(f"  {mod:14s} | {st:3s} | n=0")
                continue
            pcts = [100.0 * np.mean(v >= thr[k][st]) for k in K_VALUES]
            print(f"  {mod:14s} | {st:3s} | n={len(v):4d} "
                  f"median_eff={np.median(v):.3f} | "
                  f"clear%: k2={pcts[0]:5.1f}  k3={pcts[1]:5.1f}  "
                  f"k4={pcts[2]:5.1f}")

    print("\nThreshold reference (min ML eff to clear 0.45 tetramer):")
    for k in K_VALUES:
        print(f"  k={k}: het>={thr[k]['het']:.3f}  loh>={thr[k]['loh']:.3f}")

    print("\n" + "=" * 74)
    print("BIALLELIC ARGUMENT  (patients with >=2 TP53 mutations)")
    print("=" * 74)
    multi_pids = {p for p, n in n_muts.items() if n >= 2}
    by_patient_mods = {}
    for _, row in muts.iterrows():
        pid = row['patient_id']
        if pid not in multi_pids:
            continue
        aa = aa_from_row(row)
        if aa is None:
            continue
        effs = best_guide_efficiency(aa, ref)
        if effs:
            mod = max(effs, key=effs.get)
            by_patient_mods.setdefault(pid, set()).add(mod)
    same_mod = sum(1 for ms in by_patient_mods.values() if len(ms) == 1)
    diff_mod = sum(1 for ms in by_patient_mods.values() if len(ms) >= 2)
    tot = same_mod + diff_mod
    print(f"  biallelic patients with >=1 editable mutation: {tot}")
    if tot:
        print(f"  all mutations same modality:  {same_mod} "
              f"({100*same_mod/tot:.1f}%)  -> behaves ~ LOH (eff ceiling)")
        print(f"  mutations need diff modality: {diff_mod} "
              f"({100*diff_mod/tot:.1f}%)  -> single editor caps WT monomer "
              f"<=0.5; 0.5^k = {0.5**2:.3f}/{0.5**3:.3f}/{0.5**4:.3f} "
              f"< 0.45 -> UNRESCUABLE with one editor")

    dump = {f"{m}|{s}": grid.get((m, s), []) for m in mods for s in states}
    out_path = os.path.join(os.path.dirname(__file__), '..',
                            'data', '_rescuability_grid.json')
    with open(out_path, 'w') as f:
        json.dump({'grid': dump,
                   'thresholds': {str(k): thr[k] for k in K_VALUES}}, f)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
