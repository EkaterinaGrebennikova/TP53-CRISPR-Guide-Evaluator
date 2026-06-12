"""Allelic-state-aware therapeutic recommender (SKELETON - validate logic first).

Primary output of the tool: given a TP53 mutation and its allelic state,
returns a ranked list of therapeutic-class recommendations. CRISPR guide
design becomes ONE branch (gene correction), invoked only when relevant.

The stratification rules below are the source of truth and mirror Fig 9 /
section 3.10. THEY ENCODE SCIENTIFIC CLAIMS AND MUST BE VALIDATED before the
recommender is trusted. Evidence tags:
  'empirical'   = supported by our cancer-type-adjusted GDSC analysis
  'mechanistic' = predicted from p53 biology, not directly tested here
  'null'        = tested, no allelic-state-specific effect found
"""
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# STRATIFICATION RULES  --  source of truth (matches Fig 9 / section 3.10)
# >>> VALIDATE THIS TABLE WITH THE PROFESSOR BEFORE RELYING ON IT <<<
# level: 'recommended' | 'reduced' | 'not_recommended' | 'no_stratification'
# ---------------------------------------------------------------------------
STRATIFICATION = {
    'heterozygous_cn_neutral': {
        'MDM2_inhibitor':      ('recommended', 'mechanistic',
                                'Wild-type allele retained; MDM2i can stabilize it'),
        'dna_damaging_chemo':  ('recommended', 'mechanistic',
                                'p53-dependent apoptosis intact'),
        'antimitotic_chemo':   ('not_allelic_limited', 'mechanistic',
                                'p53-independent; efficacy not reduced by allelic '
                                'state, but selection is cancer-type-driven, not '
                                'a p53-based recommendation'),
        'synthetic_lethality': ('no_stratification', 'null',
                                'No allelic-state-specific effect after adjustment'),
        'mutant_reactivator':  ('no_stratification', 'mechanistic',
                                'Not tested as pooled LOH-vs-WT (mutation-class-'
                                'specific reactivator; see §3.7); branch refined '
                                'by mutation gate'),
        'gene_correction':     ('not_recommended', 'mechanistic',
                                'WT already present; correction not required'),
    },
    'heterozygous_with_gain': {
        'MDM2_inhibitor':      ('reduced', 'mechanistic',
                                'WT present but outnumbered by amplified mutant'),
        'dna_damaging_chemo':  ('reduced', 'mechanistic', 'Partial p53 function'),
        'antimitotic_chemo':   ('not_allelic_limited', 'mechanistic',
                                'p53-independent; not reduced by allelic state'),
        'synthetic_lethality': ('no_stratification', 'null',
                                'No allelic-state-specific effect after '
                                'cancer-type adjustment'),
        'mutant_reactivator':  ('no_stratification', 'mechanistic',
                                'Not tested as pooled LOH-vs-WT (mutation-class-'
                                'specific reactivator; see §3.7); branch refined '
                                'by mutation gate'),
        'gene_correction':     ('reduced', 'mechanistic',
                                'Adjunct: correct amplified mutant copies'),
    },
    'loh_with_mutation': {
        'MDM2_inhibitor':      ('not_recommended', 'empirical',
                                'No WT to stabilize; Nutlin-3a beta=+2.18, '
                                'p=1.7e-10 (BH within targeted panel)'),
        'dna_damaging_chemo':  ('not_recommended', 'empirical',
                                'LOH-resistant to 5 BH-significant DNA-damaging agents'),
        'antimitotic_chemo':   ('not_allelic_limited', 'empirical',
                                'p53-independent; spared (no LOH effect). Efficacy '
                                'not reduced by LOH, but not a p53-based pick'),
        'synthetic_lethality': ('no_stratification', 'null',
                                'PARP/ATR/CHK1: no effect after cancer-type adjustment'),
        'mutant_reactivator':  ('no_stratification', 'mechanistic',
                                'Not tested as pooled LOH-vs-WT (mutation-class-'
                                'specific reactivator; see §3.7); branch refined '
                                'by mutation gate'),
        'gene_correction':     ('recommended', 'empirical',
                                'Only restoration path; subject to modality-dependent '
                                'ceiling (~17-19% Phi-debiased WT-loss-attributable '
                                'non-correctability across base-editable LOH)'),
    },
    'biallelic_mutation': {
        'MDM2_inhibitor':      ('not_recommended', 'mechanistic', 'No WT'),
        'dna_damaging_chemo':  ('not_recommended', 'mechanistic',
                                'Apoptotic response lost'),
        'antimitotic_chemo':   ('not_allelic_limited', 'mechanistic',
                                'p53-independent; not reduced by allelic state'),
        'synthetic_lethality': ('no_stratification', 'null',
                                'No allelic-state-specific effect after '
                                'cancer-type adjustment'),
        'mutant_reactivator':  ('no_stratification', 'mechanistic',
                                'Not tested as pooled LOH-vs-WT (mutation-class-'
                                'specific reactivator; see §3.7); branch refined '
                                'by mutation gate'),
        'gene_correction':     ('recommended', 'mechanistic',
                                'Restoration path; multi-guide, technically hardest'),
    },
}

# representative agents per class (for the recommendation text)
CLASS_AGENTS = {
    'MDM2_inhibitor':      'Nutlin-3a, idasanutlin',
    'dna_damaging_chemo':  '5-FU, gemcitabine, cytarabine, doxorubicin, cisplatin',
    'antimitotic_chemo':   'paclitaxel, docetaxel, vinorelbine',
    'synthetic_lethality': 'olaparib (PARP), AZD6738 (ATR), AZD7762 (CHK1)',
    'mutant_reactivator':  'APR-246/eprenetapopt, COTI-2',
    'gene_correction':     'CBE / ABE / prime editing / HDR',
}

# rank order for presentation (recommended classes first)
# 'not_allelic_limited' = p53-independent; efficacy not reduced by allelic state,
# but selection is cancer-type-driven (not a p53-based recommendation).
_LEVEL_RANK = {'recommended': 0, 'reduced': 1, 'not_allelic_limited': 2,
               'no_stratification': 3, 'not_recommended': 4}


@dataclass
class TherapyRecommendation:
    modality_class: str
    level: str
    evidence: str
    rationale: str
    agents: str = ''
    correction_detail: Optional[dict] = None   # filled for gene_correction


# allelic state -> allele-model zygosity (allelemodel.get_allele_status)
_ZYGOSITY = {
    'heterozygous_cn_neutral': 'heterozygous',
    'heterozygous_with_gain':  'heterozygous',
    'loh_with_mutation':       'loh',
    'biallelic_mutation':      'homozygous',
}

VENTURA_THRESHOLD = 0.45        # functional-tetramer threshold (Ventura 2007)
_DEFAULT_EFFICIENCY = 0.70      # used when a modality has no ML model (PE/HDR)


def recommend_treatments(allelic_state: str,
                         parsed_mutation=None,
                         mutation_eval=None,
                         design_guides: bool = False,
                         cds_sequence: Optional[str] = None
                         ) -> List[TherapyRecommendation]:
    """Primary entry point. Return ranked therapy recommendations for an
    allelic state, optionally enriched with mutation-specific gene-correction
    assessment.

    Args:
        allelic_state: one of the keys in STRATIFICATION.
        parsed_mutation: optional ParsedMutation (needed for gene-correction
                         detail / guide design).
        mutation_eval: optional MutationEvaluation (reserved for future use,
                       e.g. severity-aware refinement).
        design_guides: if True and a ParsedMutation is given, run the CRISPR
                       pipeline to attach correctability detail to the
                       gene_correction recommendation.
        cds_sequence: optional TP53 CDS; fetched from reference if omitted.

    Returns:
        list[TherapyRecommendation], ranked recommended -> not_recommended.
    """
    if allelic_state not in STRATIFICATION:
        raise ValueError(
            f"Unknown allelic_state '{allelic_state}'. "
            f"Expected one of {list(STRATIFICATION)}.")

    recs = []
    for modality_class, (level, evidence, rationale) in \
            STRATIFICATION[allelic_state].items():
        rec = TherapyRecommendation(
            modality_class=modality_class,
            level=level,
            evidence=evidence,
            rationale=rationale,
            agents=CLASS_AGENTS.get(modality_class, ''),
        )
        if (modality_class == 'gene_correction' and design_guides
                and parsed_mutation is not None):
            try:
                rec.correction_detail = assess_gene_correction(
                    parsed_mutation, allelic_state, cds_sequence)
            except Exception as ex:                       # never crash the rec
                rec.correction_detail = {'error': str(ex)}
        recs.append(rec)

    if mutation_eval is not None:
        _refine_for_mutation(recs, mutation_eval)

    recs.sort(key=lambda r: (_LEVEL_RANK.get(r.level, 99), r.modality_class))
    return recs


# mutation-specific reactivators (the mutation-gate layer)
_MUTATION_SPECIFIC_REACTIVATOR = {
    'Y220C': 'rezatapopt (PC14586) is Y220C-specific (binds the Y220C cryptic '
             'pocket); reactivator class not tested as pooled LOH-vs-WT in our '
             'data (§3.7)',
}


def _is_truncating(aa_change: str) -> bool:
    s = (aa_change or '')
    return ('*' in s) or ('fs' in s) or ('del' in s.lower()) \
        or ('ins' in s.lower()) or ('splice' in s.lower())


def _refine_for_mutation(recs, mutation_eval) -> None:
    """Mutation-gate layer (scope: allelic + mutation gates). Refines the
    mutant-reactivator branch by mutation type / structural class. Efficacy
    *levels* are unchanged (allelic-state data drives those); only the
    reactivator rationale is refined to reflect whether the drug class is even
    mechanistically applicable to this mutation.
    """
    aa = getattr(mutation_eval, 'aa_change', '') or ''
    for r in recs:
        if r.modality_class != 'mutant_reactivator':
            continue
        if _is_truncating(aa):
            r.rationale = ('N/A: truncating mutation yields no targetable '
                           'mutant protein to reactivate')
            r.evidence = 'mechanistic'
        elif aa in _MUTATION_SPECIFIC_REACTIVATOR:
            r.rationale = _MUTATION_SPECIFIC_REACTIVATOR[aa]
            r.evidence = 'literature'
        elif getattr(mutation_eval, 'is_contact_residue', False):
            r.rationale = ('DNA-contact mutant; less responsive to '
                           'conformational reactivators (APR-246); reactivator '
                           'class not tested as pooled LOH-vs-WT (§3.7)')
            r.evidence = 'mechanistic'
        elif getattr(mutation_eval, 'in_dbd', False):
            r.rationale = ('structural/destabilizing DBD mutant; APR-246/'
                           'eprenetapopt mechanistically applicable; reactivator '
                           'class not tested as pooled LOH-vs-WT (§3.7)')
            r.evidence = 'mechanistic'


def assess_gene_correction(parsed_mutation, allelic_state: str,
                           cds_sequence: Optional[str] = None) -> dict:
    """Determine correctability for the gene-correction branch by invoking the
    existing CRISPR pipeline (modality selection -> guide design -> guide
    scoring/ML efficiency -> allele + tetramer model).

    Returns a dict with: best_modality, best_guide_spacer,
    best_guide_efficiency, composite_score, predicted_tetramer_fraction,
    clears_threshold (bool), note. If no scorable guide exists (e.g. HDR-only),
    returns {'correctable': False, 'note': ...}.
    """
    # lazy imports: the ML model loads on import of guidescorer/efficiencypredictor
    from mutationparser import get_reference
    from modalityselector import select_modalities
    from guidedesigner import design_guide
    from guidescorer import score_guide
    from offtargetscorer import score_offtarget
    from allelemodel import get_allele_status

    if cds_sequence is None:
        cds_sequence = get_reference().cds_sequence

    best = None            # (composite, guide, modality_label)
    best_score = -1.0
    for item in select_modalities(parsed_mutation):
        mod, nt = item['modality'], item['nt_change']
        try:
            candidates = design_guide(nt, mod, cds_sequence)
        except Exception:
            continue
        for g in candidates:
            if not g.get('spacer'):
                continue
            sg = score_guide(g, mod, nt, cds_sequence)      # sets ml_efficiency
            sot = score_offtarget(g['spacer'])
            composite = sg * 0.6 + sot * 0.4
            mod_label = 'Prime Editing' if 'pbs' in g else mod
            if composite > best_score:
                best_score = composite
                best = (g, mod_label)

    if best is None:
        return {'correctable': False,
                'note': 'No scorable base-editing/PE guide found '
                        '(likely an indel/complex change requiring HDR)'}

    guide, modality = best
    e = guide.get('ml_efficiency')                          # None for PE/HDR
    eff_for_model = e if e is not None else _DEFAULT_EFFICIENCY
    zygosity = _ZYGOSITY.get(allelic_state, 'heterozygous')
    allele = get_allele_status(zygosity, eff_for_model)
    clears = allele['passes_threshold']

    if e is None:
        note = (f"{modality} guide has no ML efficiency model; functional "
                f"estimate uses default efficiency {_DEFAULT_EFFICIENCY:.2f}")
    elif clears:
        note = (f"Predicted to restore functional p53 above the "
                f"{VENTURA_THRESHOLD} tetramer threshold")
    else:
        note = (f"Predicted below the {VENTURA_THRESHOLD} functional threshold; ")
        if zygosity in ('loh', 'homozygous') and modality == 'CBE':
            note += ("CBE correction with the wild-type allele absent faces the "
                     "correctability ceiling — consider prime editing or "
                     "combination therapy")
        else:
            note += ("single-allele correction insufficient at predicted "
                     "efficiency; consider combination or alternative modality")

    return {
        'best_modality': modality,
        'best_guide_spacer': guide.get('spacer'),
        'best_guide_efficiency': e,
        'composite_score': round(best_score, 3),
        'effective_wt_fraction': allele['effective_wt_fraction'],
        'predicted_tetramer_fraction': allele['tetramer_fraction'],
        'clears_threshold': clears,
        'note': note,
    }


# display labels for human-readable output
_LEVEL_DISPLAY = {
    'recommended':        'RECOMMENDED',
    'reduced':            'REDUCED EFFICACY',
    'not_allelic_limited': 'p53-INDEPENDENT (efficacy not allelic-state-limited)',
    'no_stratification':  'NO ALLELIC-STATE GUIDANCE',
    'not_recommended':    'NOT RECOMMENDED',
}
_CLASS_DISPLAY = {
    'MDM2_inhibitor':      'MDM2 inhibitors',
    'dna_damaging_chemo':  'DNA-damaging chemotherapy',
    'antimitotic_chemo':   'Anti-mitotic chemotherapy',
    'synthetic_lethality': 'Synthetic-lethality agents',
    'mutant_reactivator':  'Mutant-p53 reactivators',
    'gene_correction':     'CRISPR gene correction',
}


def format_recommendations(recs: List[TherapyRecommendation],
                           mutation_label: str,
                           allelic_state: str) -> str:
    """Render the primary human-readable output: ranked therapy classes first,
    with gene-correction/CRISPR detail appended only where computed.
    """
    out = []
    out.append(f"TREATMENT RECOMMENDATIONS")
    out.append(f"  Mutation:     {mutation_label}")
    out.append(f"  Allelic state: {allelic_state}")
    out.append("=" * 68)
    for i, r in enumerate(recs, 1):
        out.append(f"\n{i}. [{_LEVEL_DISPLAY.get(r.level, r.level)}]  "
                   f"{_CLASS_DISPLAY.get(r.modality_class, r.modality_class)}")
        if r.agents:
            out.append(f"     agents: {r.agents}")
        out.append(f"     basis ({r.evidence}): {r.rationale}")
        d = r.correction_detail
        if d:
            if 'error' in d:
                out.append(f"     [gene-correction assessment failed: {d['error']}]")
            elif not d.get('best_modality'):
                out.append(f"     [{d.get('note', '')}]")
            else:
                eff = d['best_guide_efficiency']
                eff_str = (f"{eff:.3f}" if eff is not None
                           else "N/A (no ML model for this modality)")
                clears = "CLEARS" if d['clears_threshold'] else "BELOW"
                out.append(f"     -> best modality: {d['best_modality']}  "
                           f"(spacer {d['best_guide_spacer']})")
                out.append(f"     -> predicted ML correction efficiency: {eff_str}")
                out.append(f"     -> predicted functional tetramer fraction: "
                           f"{d['predicted_tetramer_fraction']:.3f} "
                           f"({clears} the {VENTURA_THRESHOLD} threshold)")
                out.append(f"     -> {d['note']}")
    return "\n".join(out)


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from mutationparser import parse_mutations

    demo = [('R175H', 'loh_with_mutation'),
            ('R175H', 'heterozygous_cn_neutral'),
            ('Y220C', 'loh_with_mutation')]
    for aa, state in demo:
        parsed = parse_mutations([aa])[0]
        recs = recommend_treatments(state, parsed_mutation=parsed,
                                    design_guides=True)
        print("\n" + format_recommendations(recs, aa, state))
        print()

