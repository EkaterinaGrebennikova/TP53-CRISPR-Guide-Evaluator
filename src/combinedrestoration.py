tetramer_threshold = 0.45

def get_combined_restoration(strategies, efficiency=0.7):
    tetramer_fractions = []
    individual_scores = {}
    for s in strategies:
        tetramer_fraction = efficiency ** 4
        structural_impact = s['evaluation'].structural_impact if s['evaluation'].structural_impact is not None else 0.5
        structural_recovery = 1.0 - structural_impact
        individual_score = round(tetramer_fraction * structural_recovery, 3)
        tetramer_fractions.append(tetramer_fraction)
        name = f"{s['mutation'].ref_aa}{s['mutation'].aa_position}{s['mutation'].alt_aa}"
        individual_scores[name] = individual_score
    product = 1.0
    for fraction in tetramer_fractions:
        product = product * (1 - fraction)
    combined_tetramer = round(1 - product, 3)
    combined_score = round(sum(individual_scores.values()) / len(individual_scores), 3)
    passes = combined_tetramer >= tetramer_threshold
    return {
        'individual_scores': individual_scores,
        'combined_tetramer': combined_tetramer,
        'combined_score':    combined_score,
        'passes_threshold':  passes,
        'efficiency_used':   efficiency
    }

