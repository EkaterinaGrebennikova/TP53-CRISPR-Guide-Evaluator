import argparse, sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.mutationparser import parse_mutations
from src.mutationevaluator import evaluate_mutations
from src.strategybuilder import build_strategies
from src.offtargetscorer import score_offtarget
from src.rnaofftarget import score_rna_offtarget
from src.deliveryrecommender import recommend_delivery
from src.cellcyclewarning import get_cell_cycle
from src.allelemodel import get_allele_status
from src.aggregationflag import get_aggregation_risk
from src.iarcp53 import get_iarc_annotation, get_germline_status
from src.pathwaystatus import get_pathway_status
from src.treatmentrecommender import (
    recommend_treatments, _CLASS_DISPLAY, VENTURA_THRESHOLD,
    _ZYGOSITY as ALLELIC_TO_ZYGOSITY,
)

# user-facing allelic-state names -> internal classifier keys
ALLELIC_MAP = {
    'het':       'heterozygous_cn_neutral',
    'het_gain':  'heterozygous_with_gain',
    'loh':       'loh_with_mutation',
    'biallelic': 'biallelic_mutation',
}
# back-compat: --zygosity -> internal allelic-state key
ZYG_TO_ALLELIC = {
    'heterozygous': 'heterozygous_cn_neutral',
    'loh':          'loh_with_mutation',
    'homozygous':   'biallelic_mutation',
}
ALLELIC_DISPLAY = {
    'heterozygous_cn_neutral': 'heterozygous CN-neutral (WT retained)',
    'heterozygous_with_gain':  'heterozygous + gain',
    'loh_with_mutation':       'LOH (wild-type allele lost)',
    'biallelic_mutation':      'biallelic',
}


def _resolve_allelic_state(args) -> str:
    if args.allelic_state:
        return ALLELIC_MAP[args.allelic_state]
    if args.zygosity:
        return ZYG_TO_ALLELIC[args.zygosity]
    return 'heterozygous_cn_neutral'


# ---------------------------------------------------------------------------
# Output blocks
# ---------------------------------------------------------------------------

def _append_correction_detail(out, rec):
    d = rec.correction_detail
    if not d:
        return
    if 'error' in d:
        out.append(f"     [gene-correction assessment failed: {d['error']}]")
    elif not d.get('best_modality'):
        out.append(f"     {d.get('note', '')}")
    else:
        eff = d['best_guide_efficiency']
        eff_str = f"{eff:.3f}" if eff is not None else "N/A (no ML model)"
        clears = "CLEARS" if d['clears_threshold'] else "BELOW"
        out.append(f"     -> best modality: {d['best_modality']}  "
                   f"(guide {d['best_guide_spacer']})")
        out.append(f"     -> ML efficiency {eff_str}   "
                   f"tetramer {d['predicted_tetramer_fraction']:.3f} "
                   f"({clears} the {VENTURA_THRESHOLD} threshold)")
        if not d['clears_threshold']:
            out.append(f"     !! {d['note']}")


def format_recommendation(recs, mutation_label, allelic_state, cancer_type):
    """Recommendation-first, tiered-by-action output block."""
    by = {}
    for r in recs:
        by.setdefault(r.level, []).append(r)

    out = []
    ct = (f"  |  cancer type: {cancer_type}"
          if cancer_type and cancer_type.lower() != 'all' else '')
    line = "=" * 62
    out.append(line)
    out.append(f"  TP53 TREATMENT EVALUATOR")
    out.append(f"  {mutation_label}  |  allelic state: "
               f"{ALLELIC_DISPLAY.get(allelic_state, allelic_state)}{ct}")
    out.append(line)
    out.append("")

    if allelic_state == 'heterozygous_with_gain':
        out.append("!! NOTE: het+gain was NOT directly evaluated in our "
                   "drug-response analysis.")
        out.append("   The calls below are mechanistic expectations "
                   "(intermediate between")
        out.append("   heterozygous CN-neutral and LOH), not data-backed "
                   "recommendations.")
        out.append("")

    primary_pool = by.get('recommended', []) or by.get('reduced', [])
    if primary_pool:
        p = primary_pool[0]
        out.append(f">> PRIMARY RECOMMENDATION:  "
                   f"{_CLASS_DISPLAY.get(p.modality_class)}")
        out.append(f"     basis ({p.evidence}): {p.rationale}")
        _append_correction_detail(out, p)
        for p2 in primary_pool[1:]:
            out.append(f"\n   ALSO RECOMMENDED:  "
                       f"{_CLASS_DISPLAY.get(p2.modality_class)}")
            out.append(f"     basis ({p2.evidence}): {p2.rationale}")
            _append_correction_detail(out, p2)

    if by.get('recommended') and by.get('reduced'):
        out.append("\n   REDUCED EFFICACY")
        for r in by['reduced']:
            out.append(f"     - {_CLASS_DISPLAY.get(r.modality_class)}: "
                       f"{r.rationale}")

    if by.get('not_allelic_limited'):
        out.append("\n   CONSIDER  (p53-independent; not allelic-state-limited)")
        for r in by['not_allelic_limited']:
            out.append(f"     - {_CLASS_DISPLAY.get(r.modality_class)}  "
                       f"({r.agents})")
            out.append(f"       {r.rationale}")

    if by.get('not_recommended'):
        out.append("\n   AVOID  (allelic state predicts resistance)")
        for r in by['not_recommended']:
            out.append(f"     - {_CLASS_DISPLAY.get(r.modality_class)} -- "
                       f"{r.rationale}")
            if r.modality_class == 'gene_correction':
                _append_correction_detail(out, r)

    if by.get('no_stratification'):
        out.append("\n   NO ALLELIC-STATE GUIDANCE")
        for r in by['no_stratification']:
            out.append(f"     - {_CLASS_DISPLAY.get(r.modality_class)}: "
                       f"{r.rationale}")
    return "\n".join(out)


def print_evaluation_block(ev, tcga_freq_lookup=None, tcga_merged=None):
    print("\n" + "-" * 62)
    print("MUTATION EVALUATION")
    print(f"   Amino-acid change:    {ev.aa_change}")
    print(f"   Domain:               {ev.domain}")
    cls = []
    if ev.in_dbd and not ev.is_contact_residue:
        cls.append('structural (destabilizing)')
    if ev.is_contact_residue:
        cls.append('DNA-contact')
    if ev.iarc_hotspot or ev.is_gof:
        cls.append('hotspot' if ev.iarc_hotspot else '')
    if ev.is_gof:
        cls.append('GOF')
    cls = [c for c in cls if c]
    print(f"   Mutation class:       {' / '.join(cls) if cls else 'missense'}")
    print(f"   DNA-contact residue:  {'yes' if ev.is_contact_residue else 'no'}")
    print(f"   ClinVar:              {ev.clinvar_significance}")
    if ev.dms_score is not None:
        print(f"   DMS score:            {ev.dms_score:.3f}   ->   "
              f"functional severity {ev.functional_severity:.3f}")
    else:
        print(f"   Functional severity:  {ev.functional_severity:.3f} "
              f"(regression fallback)")
    if ev.structural_impact is not None:
        print(f"   Structural impact:    {ev.structural_impact}")
    agg = get_aggregation_risk(ev.aa_change)
    if agg['aggregates']:
        print(f"   Aggregation / DN:     {agg['note']}")
    # prevalence (best-effort from TCGA)
    if tcga_freq_lookup and ev.aa_change in tcga_freq_lookup:
        rank, entry = tcga_freq_lookup[ev.aa_change]
        line = (f"   Prevalence:           {entry['fraction']*100:.2f}% of "
                f"TP53-mutant tumors (rank #{rank})")
        if tcga_merged is not None:
            rows = tcga_merged[tcga_merged['aa_change'] == ev.aa_change]
            if not rows.empty and 'cancer_type' in rows.columns:
                top = rows['cancer_type'].dropna().value_counts().head(3)
                if not top.empty:
                    line += "; top: " + ", ".join(top.index)
        print(line)


def print_iarc_block(ev):
    iarc = get_iarc_annotation(ev.aa_change)
    if not iarc.get('found'):
        return
    print("\nIARC FUNCTIONAL ANNOTATION")
    print(f"   Transactivation:      {iarc.get('transactivation_class')}")
    print(f"   Structure/function:   {iarc.get('structure_function_class')}")
    gof = iarc.get('experimental_gof')
    print(f"   Experimental GOF:     {gof if gof else 'not reported'}")
    ts = iarc.get('temperature_sensitive')
    if ts:
        print(f"   Temperature-sensitive: {ts}")
    print(f"   Hotspot:              {iarc.get('hotspot')}")
    print(f"   Somatic count:        {iarc.get('somatic_count')}    "
          f"Germline count: {iarc.get('germline_count')}")
    germ = get_germline_status(ev.aa_change)
    if germ.get('germline_present'):
        print(f"   Germline (Li-Fraumeni): yes ({germ.get('family_count')} families)")


def print_functional_model(allelic_state, best_eff, modality_label):
    zyg = ALLELIC_TO_ZYGOSITY.get(allelic_state, 'heterozygous')
    allele = get_allele_status(zyg, best_eff)
    pre_wt = allele['pre_correction_wt_fraction']
    pre_tet = allele['pre_correction_tetramer']
    post_wt = allele['effective_wt_fraction']
    post_tet = allele['tetramer_fraction']
    print("\nFUNCTIONAL RESTORATION MODEL   "
          f"(allelic state: {ALLELIC_DISPLAY.get(allelic_state, allelic_state)})")
    print(f"                              pre-correction      "
          f"post-correction ({modality_label})")
    print(f"   WT monomer fraction        {pre_wt:<20}{post_wt}")
    print(f"   Functional tetramer        {pre_tet:<20}{post_tet}")
    pre_ok = 'meets' if pre_tet >= VENTURA_THRESHOLD else 'below'
    post_ok = 'meets' if post_tet >= VENTURA_THRESHOLD else 'below'
    print(f"   vs Ventura {VENTURA_THRESHOLD} threshold:   "
          f"{pre_ok:<20}{post_ok}")
    return allele


def print_prognosis_block(ev, cell_line, allele):
    """Cell-line-specific prognosis (only when --cell-line is supplied)."""
    pre = get_pathway_status(cell_line, ev, allele['pre_correction_wt_fraction'],
                             post_correction=False)
    post = get_pathway_status(cell_line, ev, allele['effective_wt_fraction'],
                              post_correction=True)
    print(f"\nCELL-LINE PROGNOSIS  ({cell_line})")
    print(f"   MDM2: {post['mdm2']}   p21: {post['p21']}   "
          f"BAX/PUMA: {post['bax']}/{post['puma']}")
    print(f"                              pre-correction      post-correction")
    print(f"   Restoration score          {pre['score']:<20}{post['score']}")
    print(f"   Prognosis                  {pre['prognosis']:<20}{post['prognosis']}")


def print_crispr_design(strategy, cell_line):
    """Full CRISPR guide-design detail (only with --design-guides)."""
    s, ev, g = strategy, strategy['evaluation'], strategy['best_guide']
    print("\n" + "-" * 62)
    print("CRISPR GUIDE DESIGN")
    print(f"   Modality:        {s['modality']}")
    print(f"   Guide score:     {s['score']:.3f}")
    if g and g.get('spacer'):
        print(f"   Cas9 variant:    {g.get('cas9_variant', 'N/A')}")
        print(f"   Top guide:       {g['spacer']}")
        ml = g.get('ml_efficiency')
        if ml is not None:
            print(f"   ML efficiency:   {ml:.3f}")
        print(f"   Off-target:      {score_offtarget(g['spacer'])}")
        print(f"   RNA off-target:  {score_rna_offtarget(g['spacer'], s['modality'])}")
        if 'pbs' in g:
            print(f"   PBS / RTT:       {g['pbs']} / {g['rtt']}")
    else:
        print(f"   Note:            {g.get('notes') if g else 'No guide found'}")
    if cell_line:
        cc = get_cell_cycle(s['modality'], cell_line)
        if cc:
            print(f"   [cell-cycle] {cc}")
        rec = recommend_delivery(s['modality'], cell_line)
        print(f"   Delivery:        {rec['primary']} ({rec['primary_eff']}) / "
              f"{rec['secondary']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TP53 Treatment Evaluator (allelic-state-guided)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mutations', '-m', type=str,
                       help="Comma-separated mutations, e.g. 'R175H,R248W'")
    group.add_argument('--vcf', type=str, help="Path to a VCF file")

    parser.add_argument('--allelic-state', choices=list(ALLELIC_MAP),
                        help="Primary input: het / het_gain / loh / biallelic")
    parser.add_argument('--zygosity', choices=list(ZYG_TO_ALLELIC),
                        help="(alias) heterozygous / loh / homozygous")
    parser.add_argument('--cell-line', default=None,
                        choices=['HCT116', 'U2OS', 'MCF7'],
                        help="Optional: adds cell-line-specific prognosis")
    parser.add_argument('--cancer-type', default='all',
                        help="Cancer-type context label")
    parser.add_argument('--design-guides', action='store_true',
                        help="Show the full CRISPR guide-design section")
    parser.add_argument('--landscape', action='store_true',
                        help="Show pan-cancer TCGA therapy landscape")
    parser.add_argument('--output', type=str, help="Save results (.json/.tsv)")
    args = parser.parse_args()

    mutation_strings = ([m.strip() for m in args.mutations.split(',')]
                        if args.mutations else _read_vcf(args.vcf))
    allelic_state = _resolve_allelic_state(args)

    mutations = parse_mutations(mutation_strings)
    evaluations = evaluate_mutations(mutations)
    strategies = build_strategies(mutations, evaluations)

    # best-effort TCGA prevalence lookup (cheap part only)
    tcga_freq_lookup, tcga_merged = None, None
    try:
        from src.tcgaloader import (load_mutations as _lm, load_cna as _lc,
                                     load_clinical as _lcl, merge_patient_data as _mg)
        from src.tcgafrequency import get_mutation_frequencies
        _muts = _lm()
        freqs = get_mutation_frequencies(_muts)
        tcga_freq_lookup = {e['aa_change']: (i + 1, e)
                            for i, e in enumerate(freqs)}
        tcga_merged = _mg(_muts, _lc(), _lcl())
    except Exception:
        pass

    for s in strategies:
        m, ev = s['mutation'], s['evaluation']
        label = f"{m.ref_aa}{m.aa_position}{m.alt_aa}"

        recs = recommend_treatments(allelic_state, parsed_mutation=m,
                                    mutation_eval=ev, design_guides=True)
        print("\n" + format_recommendation(recs, label, allelic_state,
                                            args.cancer_type))

        print_evaluation_block(ev, tcga_freq_lookup, tcga_merged)
        print_iarc_block(ev)

        # best-guide efficiency for the functional model
        gc = next((r for r in recs if r.modality_class == 'gene_correction'), None)
        eff = None
        modality_label = 'best guide'
        if gc and gc.correction_detail and not gc.correction_detail.get('error'):
            eff = gc.correction_detail.get('best_guide_efficiency')
            modality_label = gc.correction_detail.get('best_modality', 'best guide')
        eff = eff if eff is not None else 0.70
        allele = print_functional_model(allelic_state, eff, modality_label)

        if args.cell_line:
            print_prognosis_block(ev, args.cell_line, allele)

        if args.design_guides:
            print_crispr_design(s, args.cell_line)

    if args.landscape:
        _print_landscape()

    if args.output:
        if args.output.endswith('.json'):
            save_json(strategies, args.output)
        elif args.output.endswith('.tsv'):
            save_tsv(strategies, args.output)
        else:
            print("\n[Output] Unrecognized extension -- use .json or .tsv")


def _print_landscape():
    """Pan-cancer TCGA therapy landscape (heavy; only with --landscape)."""
    try:
        from src.tcgaloader import (load_mutations as _lm, load_cna as _lc,
                                     load_clinical as _lcl)
        from src.tcgafrequency import get_correctable_fraction
        from src.tcgaallelic import get_allelic_context
        from src.survivalanalysis import build_survival_df, cox_regression
        muts, cna, clin = _lm(), _lc(), _lcl()
    except Exception as e:
        print(f"\n[Note] TCGA landscape unavailable: {e}")
        return

    print("\n" + "=" * 62)
    print("  TCGA PAN-CANCER THERAPY LANDSCAPE")
    corr = get_correctable_fraction(muts)
    print(f"\n  CRISPR modality coverage (n={corr['total']} TP53 mutations):")
    print(f"    ABE: {corr['abe']['fraction']*100:.1f}%   "
          f"CBE: {corr['cbe']['fraction']*100:.1f}%   "
          f"PE: {corr['pe']['fraction']*100:.1f}%   "
          f"HDR: {corr['hdr']['fraction']*100:.1f}%")
    ac = get_allelic_context(muts, cna)
    tot = ac['total_tp53_mutant_patients']
    print(f"\n  Allelic state distribution (n={tot}):")
    for st, lbl in [('loh_with_mutation', 'LOH'),
                    ('biallelic_mutation', 'Biallelic'),
                    ('heterozygous_with_gain', 'Het + gain'),
                    ('heterozygous_cn_neutral', 'Het CN-neutral'),
                    ('unknown', 'Unknown')]:
        print(f"    {lbl:<16} {ac['states'][st]:>5} "
              f"({ac['states'][st]/tot*100:.1f}%)")
    sdf = build_survival_df(muts, cna, clin)
    cox = cox_regression(sdf)
    print(f"\n  Cox HR by allelic state (stratified by cancer type, "
          f"n={cox['n_observations']}):")
    for key, lbl in [('allelic_state_loh_with_mutation', 'LOH'),
                     ('allelic_state_heterozygous_with_gain', 'Het + gain'),
                     ('allelic_state_biallelic_mutation', 'Biallelic'),
                     ('allelic_state_heterozygous_cn_neutral', 'Het CN-neutral')]:
        if key in cox['summary'].index:
            row = cox['summary'].loc[key]
            sig = ' *' if row['p'] < 0.05 else ''
            print(f"    {lbl:<16} HR={row['exp(coef)']:.3f} p={row['p']:.3g}{sig}")


def save_json(strategies, path):
    out = []
    for s in strategies:
        m, ev, g = s['mutation'], s['evaluation'], s['best_guide']
        out.append({
            'mutation': f"{m.ref_aa}{m.aa_position}{m.alt_aa}",
            'domain': ev.domain, 'severity': ev.functional_severity,
            'clinvar': ev.clinvar_significance, 'gof': ev.is_gof,
            'modality': s['modality'],
            'cas9_variant': g.get('cas9_variant') if g else None,
            'guide_score': s['score'], 'spacer': g.get('spacer') if g else None,
        })
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[Output] Saved JSON to {path}")


def save_tsv(strategies, path):
    headers = ['mutation', 'domain', 'severity', 'clinvar', 'gof',
               'modality', 'cas9_variant', 'guide_score', 'spacer']
    with open(path, 'w') as f:
        f.write('\t'.join(headers) + '\n')
        for s in strategies:
            m, ev, g = s['mutation'], s['evaluation'], s['best_guide']
            row = [f"{m.ref_aa}{m.aa_position}{m.alt_aa}", ev.domain,
                   ev.functional_severity, ev.clinvar_significance, ev.is_gof,
                   s['modality'], g.get('cas9_variant', '') if g else '',
                   s['score'], g.get('spacer', '') if g else '']
            f.write('\t'.join(str(x) for x in row) + '\n')
    print(f"\n[Output] Saved TSV to {path}")


def _read_vcf(path):
    if not os.path.exists(path):
        print(f"Error: VCF file not found: {path}")
        sys.exit(1)
    lines = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 5 and parts[0] in ('17', 'chr17'):
                lines.append(line.strip())
    return lines


if __name__ == "__main__":
    main()
