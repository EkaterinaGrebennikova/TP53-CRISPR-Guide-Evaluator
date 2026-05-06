tetramer_threshold = 0.45

def get_combined_restoration(strategies, efficiency=0.7, zygosity='heterozygous'):
    from allelemodel import get_allele_status
    individual_pre  = {}
    individual_post = {}
    pre_tetramers   = []
    post_tetramers  = []

    for s in strategies:
        name = f"{s['mutation'].ref_aa}{s['mutation'].aa_position}{s['mutation'].alt_aa}"
        g = s.get('best_guide')
        eff = g.get('ml_efficiency', efficiency) if g else efficiency
        allele = get_allele_status(zygosity, eff)
        structural_impact = s['evaluation'].structural_impact if s['evaluation'].structural_impact is not None else 0.5

        pre_score  = round(allele['pre_correction_tetramer']  * (1.0 - structural_impact), 3)
        post_score = round(allele['tetramer_fraction']        * 1.0, 3)  # post: structural_recovery = 1.0

        individual_pre[name]  = pre_score
        individual_post[name] = post_score
        pre_tetramers.append(allele['pre_correction_tetramer'])
        post_tetramers.append(allele['tetramer_fraction'])

    def combined_tetramer(fractions):
        product = 1.0
        for f in fractions:
            product *= (1 - f)
        return round(1 - product, 3)

    combined_pre  = combined_tetramer(pre_tetramers)
    combined_post = combined_tetramer(post_tetramers)
    delta = round(combined_post - combined_pre, 3)
    shortfall = round(tetramer_threshold - combined_post, 3)

    contributions = {}
    total_gain = sum(individual_post[n] - individual_pre[n] for n in individual_post)
    for name in individual_post:
        gain = round(individual_post[name] - individual_pre[name], 3)
        pct  = round((gain / total_gain * 100) if total_gain > 0 else 0, 1)
        contributions[name] = {'gain': gain, 'pct': pct}

    return {
        'individual_pre':    individual_pre,
        'individual_post':   individual_post,
        'contributions':     contributions,
        'combined_pre':      combined_pre,
        'combined_post':     combined_post,
        'delta':             delta,
        'shortfall':         shortfall if shortfall > 0 else 0,
        'passes_threshold':  combined_post >= tetramer_threshold,
        'efficiency_used':   efficiency,
        'zygosity':          zygosity
    }

