import argparse, sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.mutationparser import parse_mutations
from src.mutationevaluator import evaluate_mutations
from src.strategybuilder import build_strategies
from src.offtargetscorer import score_offtarget
from src.rnaofftarget import score_rna_offtarget
from src.deliveryrecommender import recommend_delivery
from src.pathwaystatus import get_pathway_status
from src.tetramodel import get_tetramer_status
from src.transcriptionaltargets import get_transcriptional_restoration
from src.guidesetoptimizer import optimize_guide_set
from src.combinedrestoration import get_combined_restoration
from src.aggregationflag import get_aggregation_risk
from src.cellcyclewarning import get_cell_cycle
from src.therapeuticwindow import get_therapeutic_window
from src.compoundtherapy import get_compound_synergy
from src.allelemodel import get_allele_status
from src.iarcp53 import get_iarc_annotation, get_germline_status

def main():
    parser  = argparse.ArgumentParser(description="TP53 CRISPR Correction Optimizer")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mutations', '-m', type=str,
                       help="Comma-separated mutations, e.g. 'R175H,R248W'")
    group.add_argument('--vcf', type=str,
                       help="Path to a VCF file")

    parser.add_argument('--cell-line', default='HCT116',
                        choices=['HCT116', 'U2OS', 'MCF7'])
    parser.add_argument('--zygosity', default='heterozygous',
                        choices=['heterozygous', 'homozygous', 'loh'],
                        help="Allelic status of the TP53 mutation")
    parser.add_argument('--cancer-type', default='all',
                        help="Cancer type filter (e.g. breast, lung, colorectal, all)")
    parser.add_argument('--top', type=int, default=3)
    parser.add_argument('--show-all-guides', action='store_true')
    parser.add_argument('--output', type=str, help="Save results to file (.json or .tsv)")
    parser.add_argument('--summary', action='store_true', help="Print unified correction strategy summary")

    args = parser.parse_args()

    if args.mutations:
        mutation_strings = [m.strip() for m in args.mutations.split(',')]
    else:
        mutation_strings = _read_vcf(args.vcf)

    print(f"\n=== TP53 CRISPR Correction Optimizer ===")
    print(f"Mutations: {', '.join(mutation_strings)}  |  Cell line: {args.cell_line}  |  Zygosity: {args.zygosity}  |  Cancer type: {args.cancer_type}\n")

    mutations = parse_mutations(mutation_strings)
    evaluations = evaluate_mutations(mutations)
    strategies = build_strategies(mutations, evaluations)

    for s in strategies:
        m  = s["mutation"]
        ev = s["evaluation"]
        g  = s["best_guide"]
        print(f"\n{'='*50}")
        print(f"  Mutation:         {m.ref_aa}{m.aa_position}{m.alt_aa}")
        print(f"  Domain:           {ev.domain}")
        print(f"  Severity score:   {ev.functional_severity:.2f}")
        print(f"  ClinVar:          {ev.clinvar_significance}")
        print(f"  GOF:              {ev.is_gof}")
        print(f"  Driver genes:     {', '.join(ev.driver_genes) if ev.driver_genes else 'None'}")
        agg = get_aggregation_risk(ev.aa_change)
        if agg['aggregates']:
            print(f"  [WARNING] Aggregation: {agg['note']}")
        allele = get_allele_status(args.zygosity, 0.7)
        print(f"\n  --- Allele Model ---")
        print(f"  Zygosity:                 {allele['zygosity']}")
        print(f"  Pre-correction WT:        {allele['pre_correction_wt_fraction']}  (tetramer: {allele['pre_correction_tetramer']})")
        print(f"  Post-correction WT:       {allele['effective_wt_fraction']}  (tetramer: {allele['tetramer_fraction']})")
        if allele['passes_threshold']:
            print(f"  Threshold (Ventura 2007): PASSES  ({allele['tetramer_fraction']} >= 0.450)")
        else:
            print(f"  Threshold (Ventura 2007): [WARNING] FAILS  ({allele['tetramer_fraction']} < 0.450)")
        gap = round(0.45 - allele['tetramer_fraction'], 3)
        if gap > 0:
            print(f"  Gap to threshold:         -{gap}")
        print(f"  Note: {allele['note']}")
        tetramer = get_tetramer_status(allele['effective_wt_fraction'])
        pre_tetramer = get_tetramer_status(allele['pre_correction_wt_fraction'])
        pathway_pre  = get_pathway_status(args.cell_line, ev, pre_tetramer['fully_wt_fraction'], post_correction=False)
        pathway_post = get_pathway_status(args.cell_line, ev, tetramer['fully_wt_fraction'],     post_correction=True)
        print(f"\n  --- Cell Line: {args.cell_line} ---")
        print(f"  MDM2 status:       {pathway_post['mdm2']}")
        print(f"  p21 pathway:       {pathway_post['p21']}")
        print(f"  BAX/PUMA:          {pathway_post['bax']}/{pathway_post['puma']}")
        print(f"  Pathway competency: {pathway_post['score']}")
        print(f"                     Pre-correction    Post-correction")
        print(f"  Restoration score: {pathway_pre['score']:<18}{pathway_post['score']}")
        pre_prog  = pathway_pre['prognosis']
        post_prog = pathway_post['prognosis']
        if post_prog == 'Poor':
            print(f"  [WARNING] Prognosis: {pre_prog:<16}{post_prog}")
        else:
            print(f"  Prognosis:         {pre_prog:<18}{post_prog}")
        targets_pre  = get_transcriptional_restoration(ev, args.cell_line, pre_tetramer['fully_wt_fraction'],  post_correction=False)
        targets_post = get_transcriptional_restoration(ev, args.cell_line, tetramer['fully_wt_fraction'],      post_correction=True)
        print(f"\n  --- Transcriptional Target Restoration ---")
        print(f"                     Pre-correction    Post-correction")
        print(f"  p21  (cell cycle): {targets_pre['p21']:<18}{targets_post['p21']}")
        print(f"  MDM2 (autoreg):    {targets_pre['MDM2']:<18}{targets_post['MDM2']}")
        print(f"  PUMA (apoptosis):  {targets_pre['PUMA']:<18}{targets_post['PUMA']}")
        print(f"  BAX  (apoptosis):  {targets_pre['BAX']:<18}{targets_post['BAX']}")
        print(f"\n  --- CRISPR Strategy ---")
        print(f"  Modality:         {s['modality']}")
        cc_warning = get_cell_cycle(s['modality'], args.cell_line)
        if cc_warning:
            print(f"  [WARNING] Cell cycle: {cc_warning}")
        rec = recommend_delivery(s['modality'], args.cell_line)
        print(f"  Delivery:         {rec['primary']} ({rec['primary_eff']})")
        print(f"  Alt. delivery:    {rec['secondary']} ({rec['secondary_eff']})")
        print(f"  Cas9 variant:     {g.get('cas9_variant', 'N/A') if g else 'N/A'}")
        print(f"  Guide score:      {s['score']}")
        print(f"  Structural impact: {ev.structural_impact}")
        if g and g.get("spacer"):
            print(f"  Off-target safety: {score_offtarget(g['spacer'])}")
            print(f"  RNA off-target:    {score_rna_offtarget(g['spacer'], s['modality'])}")
            print(f"  Spacer (5'->3'):  {g['spacer']}")
            if "pbs" in g:
                print(f"  PBS:              {g['pbs']}")
                print(f"  RTT:              {g['rtt']}")
        else:
            print(f"  Note:             {g.get('notes') if g else 'No guide found'}")
        print(f"\n  --- Tetramer Model ---")
        print(f"  Assumed editing efficiency:  {tetramer['efficiency']}")
        print(f"  Fully wt tetramer fraction:  {tetramer['fully_wt_fraction']}")
        print(f"  Dominant negative risk:      {tetramer['dn_risk']}")
        if tetramer['passes_threshold']:
            print(f"  Threshold (Ventura 2007):    PASSES ({tetramer['fully_wt_fraction']} >= {tetramer['threshold']})")
        else:
            print(f"  Threshold (Ventura 2007):    [WARNING] FAILS ({tetramer['fully_wt_fraction']} < {tetramer['threshold']})")
        window = get_therapeutic_window(0.7, ev.structural_impact)
        print(f"\n  --- Therapeutic Window ---")
        print(f"  Min efficiency needed:  {window['min_efficiency']}")
        print(f"  Current efficiency:     {window['current_efficiency']}  (placeholder)")
        print(f"  Effective restoration:  {window['effective_restoration']}")
        print(f"  Recommendation:         {window['recommendation']}")
        compounds = get_compound_synergy(
            ev.aa_change, agg['aggregates'],
            tetramer['passes_threshold'], pathway_post['mdm2'], ev.is_gof
        )
        if compounds:
            print(f"\n  --- Combination Therapy Recommendations ---")
            for c in compounds:
                print(f"  {c['name']}: {c['mechanism']}")
                print(f"    Rationale: {c['reason']}")
        iarc = get_iarc_annotation(ev.aa_change)
        if iarc.get('found'):
            print(f"\n  --- IARC TP53 Annotation ---")
            print(f"  Transactivation class:  {iarc['transactivation_class']}")
            print(f"  Structure/function:     {iarc['structure_function_class']}")
            print(f"  Hotspot:                {iarc['hotspot']}")
            print(f"  Somatic count (IARC):   {iarc['somatic_count']}")
            print(f"  Germline count (IARC):  {iarc['germline_count']}")
            if iarc['experimental_gof']:
                print(f"  Experimental GOF:       {iarc['experimental_gof']}")
            if iarc['temperature_sensitive']:
                print(f"  Temperature sensitive:  {iarc['temperature_sensitive']}")
            if iarc['yeast_waf1'] is not None:
                print(f"  Yeast activity (% WT):  WAF1/p21={iarc['yeast_waf1']}  MDM2={iarc['yeast_mdm2']}  BAX={iarc['yeast_bax']}  PUMA={iarc['yeast_puma']}")
            if iarc['top_cancer_types']:
                cancers = iarc['top_cancer_types']
                top_str = ', '.join(f"{c['cancer']} {c['fraction']*100:.1f}%" for c in cancers)
                print(f"  Top cancer types:       {top_str}")
            germline = get_germline_status(ev.aa_change)
            if germline['germline_present']:
                print(f"  Germline (Li-Fraumeni): Yes  ({germline['family_count']} families)")
            else:
                print(f"  Germline (Li-Fraumeni): No")

        if args.show_all_guides:
            print(f"\n  --- All Candidate Guides ({len(s['all_guides'])}) ---")
            for ag in sorted(s['all_guides'], key=lambda x: x['score'], reverse=True):
                print(f"  [{ag['score']}] {ag['modality']} | cas9: {ag['guide'].get('cas9_variant','?')} | {ag['guide'].get('spacer')}")

    if len(strategies) > 1:
        guide_set = optimize_guide_set(strategies)
        print(f"\n{'='*50}")
        print(f"  GUIDE SET COMPATIBILITY")
        print(f"  Compatible:       {guide_set['compatible']}")
        print(f"  Co-deliverable:   {guide_set['co_delivery']}")
        print(f"  Modalities:       {', '.join(guide_set['modalities'])}")
        print(f"  Delivery groups:")
        for i, group in enumerate(guide_set['delivery_groups']):
            label = 'co-deliver' if i == 0 else 'sequential'
            print(f"    Group {i+1} ({label}): {', '.join(group)}")
        if guide_set['conflicts']:
            for c in guide_set['conflicts']:
                print(f"  [CONFLICT] {c}")
        else:
            print(f"  No conflicts detected.")

        combined = get_combined_restoration(strategies, efficiency=0.7, zygosity=args.zygosity)
        print(f"\n{'='*50}")
        print(f"  COMBINED p53 RESTORATION")
        print(f"                              Pre-correction    Post-correction")
        for name in combined['individual_pre']:
            pre  = combined['individual_pre'][name]
            post = combined['individual_post'][name]
            print(f"    {name:<28}{pre:<18}{post}")
        print(f"  Combined tetramer fraction: {combined['combined_pre']:<18}{combined['combined_post']}")
        if combined['passes_threshold']:
            print(f"  Threshold:                  FAILS             PASSES  ({combined['combined_post']} >= 0.450)")
        else:
            print(f"  Threshold:                  FAILS             [WARNING] FAILS  ({combined['combined_post']} < 0.450)")
        print(f"  Delta (correction gain):    +{combined['delta']}")
        for name, c in combined['contributions'].items():
            print(f"    {name} contribution:  +{c['gain']}  ({c['pct']}% of total gain)")
        if combined['shortfall'] > 0:
            print(f"  Shortfall to threshold:     -{combined['shortfall']}  ({'near-threshold -- combination therapy may bridge gap' if combined['shortfall'] < 0.05 else 'combination therapy required'})")

    if args.summary:
        print_summary(strategies, args.cell_line)

    if args.output:
        if args.output.endswith('.json'):
            save_json(strategies, args.output)
        elif args.output.endswith('.tsv'):
            save_tsv(strategies, args.output)
        else:
            print(f"\n[Output] Unrecognized file extension — use .json or .tsv")

def print_summary(strategies, cell_line):
    from collections import defaultdict
    print(f"\n{'='*50}")
    print(f"  CORRECTION STRATEGY SUMMARY")
    print(f"  Cell line: {cell_line}")
    print(f"  Mutations analyzed: {len(strategies)}")
    print(f"{'='*50}")

    sorted_s = sorted(strategies, key=lambda x: x['score'], reverse=True)

    print(f"\n  Priority order (by correction viability):")
    for i, s in enumerate(sorted_s):
        m = s['mutation']
        label = '[RECOMMENDED]'    if s['score'] >= 0.7 else \
                '[LOW CONFIDENCE]' if s['score'] <  0.5 else ''
        cas9 = s['best_guide'].get('cas9_variant', '?') if s['best_guide'] else '?'
        print(f"  {i+1}. {m.ref_aa}{m.aa_position}{m.alt_aa}"
              f"  {s['modality']}, {cas9},"
              f"  score {s['score']}  {label}")

    modality_groups = defaultdict(list)
    for s in strategies:
        modality_groups[s['modality']].append(
            f"{s['mutation'].ref_aa}{s['mutation'].aa_position}{s['mutation'].alt_aa}"
        )

    print(f"\n  Co-delivery grouping:")
    for modality, muts in modality_groups.items():
        print(f"  {modality}: {', '.join(muts)} -- can be co-delivered")

    low = [s for s in strategies if s['score'] < 0.5]
    if low:
        print(f"\n  [WARNING] Low confidence corrections:")
        for s in low:
            m = s['mutation']
            print(f"  {m.ref_aa}{m.aa_position}{m.alt_aa} -- score {s['score']}, consider alternative guide or PE")
    else:
        print(f"\n  All mutations above confidence threshold.")

    print(f"\n  Recommended correction order: {sorted_s[0]['mutation'].ref_aa}{sorted_s[0]['mutation'].aa_position}{sorted_s[0]['mutation'].alt_aa} first (highest viability)")

def save_json(strategies, path):
    out = []
    for s in strategies:
        m, ev, g = s['mutation'], s['evaluation'], s['best_guide']
        out.append({
            'mutation':      f"{m.ref_aa}{m.aa_position}{m.alt_aa}",
            'domain':        ev.domain,
            'severity':      ev.functional_severity,
            'clinvar':       ev.clinvar_significance,
            'gof':           ev.is_gof,
            'driver_genes':  ev.driver_genes,
            'modality':      s['modality'],
            'cas9_variant':  g.get('cas9_variant') if g else None,
            'guide_score':   s['score'],
            'spacer':        g.get('spacer') if g else None,
            'pbs':           g.get('pbs') if g else None,
            'rtt':           g.get('rtt') if g else None,
        })
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[Output] Saved JSON to {path}")

def save_tsv(strategies, path):
    headers = ['mutation', 'domain', 'severity', 'clinvar', 'gof', 'driver_genes',
               'modality', 'cas9_variant', 'guide_score', 'spacer', 'pbs', 'rtt']
    with open(path, 'w') as f:
        f.write('\t'.join(headers) + '\n')
        for s in strategies:
            m, ev, g = s['mutation'], s['evaluation'], s['best_guide']
            row = [
                f"{m.ref_aa}{m.aa_position}{m.alt_aa}",
                ev.domain,
                ev.functional_severity,
                ev.clinvar_significance,
                ev.is_gof,
                ';'.join(ev.driver_genes),
                s['modality'],
                g.get('cas9_variant', '') if g else '',
                s['score'],
                g.get('spacer', '') if g else '',
                g.get('pbs', '') if g else '',
                g.get('rtt', '') if g else '',
            ]
            f.write('\t'.join(str(x) for x in row) + '\n')
    print(f"\n[Output] Saved TSV to {path}")

def _read_vcf(path):
    if not os.path.exists(path):
        print(f"Error: VCF file not found: {path}")
        sys.exit(1)

    lines = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):       # skip header lines
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            chrom = parts[0]
            if chrom in ('17', 'chr17'):   # only keep TP53 variants
                lines.append(line.strip())
    return lines


if __name__ == "__main__":
    main()
