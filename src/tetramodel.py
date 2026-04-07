def calc_tetramer_fraction(efficiency):
    return round(efficiency ** 4, 3)

def calc_dom_neg_risk(efficiency):
    return round(1 - efficiency ** 4, 3)

def get_tetramer_status(efficiency):
    fully_wt = calc_tetramer_fraction(efficiency)
    dn_risk = calc_dom_neg_risk(efficiency)
    func_threshold = 0.45
    passes = fully_wt >= func_threshold
    return {
        'efficiency': efficiency,
        'fully_wt_fraction': fully_wt,
        'dn_risk': dn_risk,
        'passes_threshold': passes,
        'threshold': func_threshold
    }