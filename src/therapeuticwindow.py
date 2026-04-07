tetramer_threshold = 0.45

def calc_min_efficiency(threshold = tetramer_threshold):
    return threshold ** (1/4)

def get_therapeutic_window(current_efficiency, structural_impact):
    min_eff = calc_min_efficiency()
    structural_recovery = 1.0 - structural_impact
    effective_restoration = current_efficiency ** 4 * structural_recovery
    viable = current_efficiency >= min_eff
    gap = min_eff - current_efficiency
    if viable:
        recommendation = 'Viable - correction expected to be therapeutic'
    elif gap < 0.1:
        recommendation = 'Increase delivery efficiency - within 10% of threshold'
    else:
        recommendation = 'Combination therapy required - consider Nutlin-3 or APR-246'
    return {
        'min_efficiency': min_eff,
        'current_efficiency': current_efficiency,
        'viable': viable,
        'gap': gap,
        'effective_restoration': effective_restoration,
        'recommendation': recommendation
    }