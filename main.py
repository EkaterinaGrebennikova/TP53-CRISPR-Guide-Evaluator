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
from src.tcgaloader import load_mutations as tcga_load_mutations
from src.tcgaloader import load_cna as tcga_load_cna
from src.tcgaloader import load_clinical as tcga_load_clinical
from src.tcgaloader import merge_patient_data as tcga_merge
from src.tcgafrequency import get_mutation_frequencies, get_correctable_fraction
from src.tcgamdm2 import get_mdm2_amplification_stats, get_therapy_candidates_by_cancer_type, get_nutlin_candidate_fraction
from src.tcgaallelic import get_allelic_context, get_vaf_distribution, get_allelic_context_by_cancer_type
from src.survivalanalysis import build_survival_df, km_tp53_mut_vs_wt, km_by_allelic_state, cox_regression, survival_by_cancer_type

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

    try:
        tcga_muts = tcga_load_mutations()
        tcga_cna_df = tcga_load_cna()
        tcga_clin = tcga_load_clinical()
        tcga_merged = tcga_merge(tcga_muts, tcga_cna_df, tcga_clin)
        tcga_freqs = get_mutation_frequencies(tcga_muts)
        tcga_freq_lookup = {entry['aa_change']: (i + 1, entry) for i, entry in enumerate(tcga_freqs)}
        tcga_nutlin = get_nutlin_candidate_fraction(tcga_muts, tcga_cna_df)
        tcga_nutlin_lookup = {r['aa_change']: r for r in tcga_nutlin}
        tcga_allelic = get_allelic_context(tcga_muts, tcga_cna_df)
        tcga_allelic_by_cancer = get_allelic_context_by_cancer_type(tcga_muts, tcga_cna_df, tcga_clin)
        tcga_survival_df = build_survival_df(tcga_muts, tcga_cna_df, tcga_clin)
        tcga_km_overall = km_tp53_mut_vs_wt(tcga_survival_df)
        tcga_km_allelic = km_by_allelic_state(tcga_survival_df)
        tcga_cox = cox_regression(tcga_survival_df)
        tcga_surv_by_cancer = survival_by_cancer_type(tcga_survival_df)
        tcga_total = tcga_muts['patient_id'].nunique()
        tcga_available = True
    except Exception as e:
        print(f"[Note] TCGA data not available: {e}")
        tcga_available = False

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
        editing_eff = g.get('ml_efficiency', 0.7) if g else 0.7
        allele = get_allele_status(args.zygosity, editing_eff)
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
        print(f"  Guide score:      {s['score']:.3f}")
        ml_eff = g.get('ml_efficiency') if g else None
        if ml_eff is not None:
            print(f"  ML efficiency:    {ml_eff:.3f}")
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
        current_eff = g.get('ml_efficiency', 0.7) if g else 0.7
        window = get_therapeutic_window(current_eff, ev.structural_impact)
        print(f"\n  --- Therapeutic Window ---")
        print(f"  Min efficiency needed:  {window['min_efficiency']:.3f}")
        print(f"  Current efficiency:     {window['current_efficiency']:.3f}{'  (ML-predicted)' if g and g.get('ml_efficiency') is not None else '  (default)'}")
        print(f"  Effective restoration:  {window['effective_restoration']:.3f}")
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

        if tcga_available:
            print(f"\n  --- TCGA Pan-Cancer Frequency ---")
            if ev.aa_change in tcga_freq_lookup:
                rank, entry = tcga_freq_lookup[ev.aa_change]
                print(f"  TCGA frequency:    {entry['count']}/{tcga_total} ({entry['fraction']*100:.2f}%) -- rank #{rank} pan-cancer")
                tcga_rows = tcga_merged[tcga_merged['aa_change'] == ev.aa_change]
                if not tcga_rows.empty and 'cancer_type' in tcga_rows.columns:
                    cancer_counts = tcga_rows['cancer_type'].dropna().value_counts().head(5)
                    if not cancer_counts.empty:
                        top_str = ', '.join(f"{c} ({n})" for c, n in cancer_counts.items())
                        print(f"  Top cancer types:  {top_str}")
                if ev.aa_change in tcga_nutlin_lookup:
                    nc = tcga_nutlin_lookup[ev.aa_change]
                    flag = ' [strong Nutlin candidate]' if nc['strong_candidate'] else ''
                    print(f"  MDM2 co-amp rate:  {nc['mdm2_amp_count']}/{nc['total_patients']} ({nc['mdm2_amp_fraction']*100:.1f}%){flag}")
                vaf_dist = get_vaf_distribution(tcga_muts, ev.aa_change)
                if vaf_dist['n'] > 0:
                    print(f"  VAF distribution:  mean={vaf_dist['mean_vaf']}, median={vaf_dist['median_vaf']}, VAF>0.7 in {vaf_dist['vaf_gt_0.7_fraction']*100:.1f}%")
            else:
                print(f"  Not observed in TCGA PanCancer Atlas (n={tcga_total})")

        if args.show_all_guides:
            print(f"\n  --- All Candidate Guides ({len(s['all_guides'])}) ---")
            for ag in sorted(s['all_guides'], key=lambda x: x['score'], reverse=True):
                ml_val = ag['guide'].get('ml_efficiency')
                ml_str = f" | ml: {ml_val:.3f}" if ml_val is not None else ""
                print(f"  [{ag['score']}] {ag['modality']} | cas9: {ag['guide'].get('cas9_variant','?')} | {ag['guide'].get('spacer')}{ml_str}")

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

    if tcga_available:
        print(f"\n{'='*50}")
        print(f"  TCGA THERAPY LANDSCAPE")
        cohort_stats = get_mdm2_amplification_stats(tcga_cna_df, tcga_muts)
        therapy_by_cancer = get_therapy_candidates_by_cancer_type(tcga_cna_df, tcga_muts, tcga_clin)
        correctable = get_correctable_fraction(tcga_muts)

        print(f"\n  --- Pan-Cancer (n={cohort_stats['total_patients']}) ---")
        a = cohort_stats['cooccurrence']['tp53mut_and_mdm2amp']
        b = cohort_stats['cooccurrence']['tp53mut_only']
        c = cohort_stats['cooccurrence']['mdm2amp_only']
        d = cohort_stats['cooccurrence']['neither']
        n = cohort_stats['total_patients']
        be = b
        print(f"  Base editing candidates:    {be:>5d}  ({be/n*100:.1f}%)")
        print(f"  MDM2 inhibitor candidates:  {c:>5d}  ({c/n*100:.1f}%)")
        print(f"  Compound therapy candidates:{a:>5d}  ({a/n*100:.1f}%)")
        print(f"  No p53 intervention:        {d:>5d}  ({d/n*100:.1f}%)")
        print(f"  TP53/MDM2 mutual exclusivity: OR={cohort_stats['odds_ratio']}, Fisher p={cohort_stats['fisher_p_value']:.2e}")

        print(f"\n  --- CRISPR Modality Coverage (n={correctable['total']} TP53 mutations) ---")
        print(f"  ABE (C>T, G>A reversal): {correctable['abe']['count']:>5d} ({correctable['abe']['fraction']*100:.1f}%)")
        print(f"  CBE (T>C, A>G reversal): {correctable['cbe']['count']:>5d} ({correctable['cbe']['fraction']*100:.1f}%)")
        print(f"  Prime editing (other):   {correctable['pe']['count']:>5d} ({correctable['pe']['fraction']*100:.1f}%)")
        print(f"  HDR (indels/complex):    {correctable['hdr']['count']:>5d} ({correctable['hdr']['fraction']*100:.1f}%)")
        print(f"  Base-editable total:     {correctable['base_editable']['count']:>5d} ({correctable['base_editable']['fraction']*100:.1f}%)")

        print(f"\n  --- Allelic Context (n={tcga_allelic['total_tp53_mutant_patients']} TP53-mut patients) ---")
        states = tcga_allelic['states']
        tot_ac = tcga_allelic['total_tp53_mutant_patients']
        print(f"  Biallelic mutation:       {states['biallelic_mutation']:>5d} ({states['biallelic_mutation']/tot_ac*100:.1f}%)")
        print(f"  LOH with mutation:        {states['loh_with_mutation']:>5d} ({states['loh_with_mutation']/tot_ac*100:.1f}%)")
        print(f"  Heterozygous + gain:      {states['heterozygous_with_gain']:>5d} ({states['heterozygous_with_gain']/tot_ac*100:.1f}%)")
        print(f"  Heterozygous CN-neutral:  {states['heterozygous_cn_neutral']:>5d} ({states['heterozygous_cn_neutral']/tot_ac*100:.1f}%)")
        print(f"  Unknown/ambiguous:        {states['unknown']:>5d} ({states['unknown']/tot_ac*100:.1f}%)")
        print(f"  High-confidence LOH fraction: {tcga_allelic['loh_fraction']} (conservative lower bound; see Donehower et al. 2019)")

        print(f"\n  --- Top Mutations by MDM2 Co-amplification (min 20 patients) ---")
        ranked = [r for r in tcga_nutlin if r['total_patients'] >= 20][:10]
        print(f"  {'Mutation':12s} {'N':>5s} {'AmpN':>5s} {'Frac':>7s}")
        for r in ranked:
            print(f"  {r['aa_change']:12s} {r['total_patients']:>5d} {r['mdm2_amp_count']:>5d} {r['mdm2_amp_fraction']*100:>6.1f}%")

        print(f"\n  --- Survival Analysis (n={tcga_km_overall['n_mut'] + tcga_km_overall['n_wt']} patients with OS data) ---")
        print(f"  TP53-MUT vs WT:   median OS {tcga_km_overall['median_os_mut']:.1f} vs {tcga_km_overall['median_os_wt']:.1f} months  (logrank p={tcga_km_overall['logrank_p']:.2e})")
        print(f"  N mut / N wt:     {tcga_km_overall['n_mut']} / {tcga_km_overall['n_wt']}")
        print(f"\n  Median OS by allelic state (vs wildtype):")
        state_order = ['wildtype', 'heterozygous_cn_neutral', 'heterozygous_with_gain', 'loh_with_mutation', 'biallelic_mutation']
        for state in state_order:
            if state not in tcga_km_allelic:
                continue
            r = tcga_km_allelic[state]
            med = f"{r['median_os']:.1f}" if r['median_os'] != float('inf') else 'NR'
            p_str = f"p={r['logrank_p_vs_wt']:.2e}" if r['logrank_p_vs_wt'] is not None else '(reference)'
            print(f"    {state:<26} n={r['n']:>5}  median OS={med:>6} mo  {p_str}")
        print(f"\n  Cox regression (stratified by cancer type, n={tcga_cox['n_observations']}, C-index={tcga_cox['concordance']:.3f}):")
        cox_rows = [
            ('age', 'Age (per year)'),
            ('sex_binary', 'Sex (Male)'),
            ('allelic_state_heterozygous_cn_neutral', 'Het, CN-neutral'),
            ('allelic_state_heterozygous_with_gain', 'Het + Gain'),
            ('allelic_state_loh_with_mutation', 'LOH + Mutation'),
            ('allelic_state_biallelic_mutation', 'Biallelic Mutation'),
        ]
        for key, label in cox_rows:
            if key not in tcga_cox['summary'].index:
                continue
            row = tcga_cox['summary'].loc[key]
            hr = row['exp(coef)']
            lo = row['exp(coef) lower 95%']
            hi = row['exp(coef) upper 95%']
            p = row['p']
            sig = ' *' if p < 0.05 else ''
            print(f"    {label:<22} HR={hr:.3f}  [{lo:.3f}-{hi:.3f}]  p={p:.3g}{sig}")
        print(f"\n  TP53-MUT survival impact by cancer type (sorted by logrank p, top 10):")
        ranked_ct = sorted(tcga_surv_by_cancer.items(), key=lambda x: x[1]['logrank_p'])[:10]
        for ct, r in ranked_ct:
            mm = f"{r['median_os_mut']:.1f}" if r['median_os_mut'] != float('inf') else 'NR'
            mw = f"{r['median_os_wt']:.1f}" if r['median_os_wt'] != float('inf') else 'NR'
            sig = ' *' if r['logrank_p'] < 0.05 else ''
            print(f"    {ct:<6} n_mut={r['n_mut']:>4} n_wt={r['n_wt']:>4}  median OS mut/wt = {mm}/{mw} mo  p={r['logrank_p']:.2e}{sig}")

        if args.cancer_type and args.cancer_type.lower() != 'all':
            ct_key = args.cancer_type.upper()
            if ct_key in therapy_by_cancer:
                ct = therapy_by_cancer[ct_key]
                ct_total = sum(ct.values())
                print(f"\n  --- {ct_key}-specific (n={ct_total}) ---")
                print(f"  Base editing candidates:    {ct['base_editing']:>5d}  ({ct['base_editing']/ct_total*100:.1f}%)")
                print(f"  MDM2 inhibitor candidates:  {ct['mdm2_inhibitor']:>5d}  ({ct['mdm2_inhibitor']/ct_total*100:.1f}%)")
                print(f"  Compound therapy candidates:{ct['compound']:>5d}  ({ct['compound']/ct_total*100:.1f}%)")
                print(f"  No p53 intervention:        {ct['none']:>5d}  ({ct['none']/ct_total*100:.1f}%)")
                if ct_key in tcga_allelic_by_cancer:
                    ac = tcga_allelic_by_cancer[ct_key]
                    print(f"  LOH fraction ({ct_key}):     {ac['loh_fraction']}")
                if ct_key in tcga_surv_by_cancer:
                    sr = tcga_surv_by_cancer[ct_key]
                    mm = f"{sr['median_os_mut']:.1f}" if sr['median_os_mut'] != float('inf') else 'NR'
                    mw = f"{sr['median_os_wt']:.1f}" if sr['median_os_wt'] != float('inf') else 'NR'
                    print(f"  TP53-MUT vs WT OS:          {mm} vs {mw} mo  (logrank p={sr['logrank_p']:.2e})")
            else:
                avail = ', '.join(sorted(therapy_by_cancer.keys()))
                print(f"\n  [Note] Cancer type '{args.cancer_type}' not found in TCGA. Available: {avail}")

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
            if line.startswith('#'):       
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            chrom = parts[0]
            if chrom in ('17', 'chr17'): 
                lines.append(line.strip())
    return lines


if __name__ == "__main__":
    main()
