domain_penalties = {
    'p21':  {'DNA-binding': 0.6, 'Transactivation 1': 0.1, 'Tetramerization': 0.3, 'default': 0.2},
    'MDM2': {'DNA-binding': 0.1, 'Transactivation 1': 0.7, 'Tetramerization': 0.1, 'default': 0.1},
    'PUMA': {'DNA-binding': 0.7, 'Transactivation 1': 0.2, 'Tetramerization': 0.5, 'default': 0.3},
    'BAX':  {'DNA-binding': 0.6, 'Transactivation 1': 0.1, 'Tetramerization': 0.3, 'default': 0.2},
}

cell_line_competency = {
    'HCT116': {'p21': 1.0, 'MDM2': 1.0, 'PUMA': 1.0, 'BAX': 1.0},
    'U2OS':   {'p21': 1.0, 'MDM2': 0.8, 'PUMA': 1.0, 'BAX': 1.0},
    'MCF7':   {'p21': 0.5, 'MDM2': 0.2, 'PUMA': 1.0, 'BAX': 1.0},
}

def get_domain_penalty(target, domain):
    if domain in domain_penalties[target]:
        return domain_penalties[target][domain]
    return domain_penalties[target]['default']

def score_target(dms_score, domain_penalty, cell_competency, tetramer_fraction):
    if dms_score is None:
        dms_score = 0.5
    return round(dms_score * (1.0 - domain_penalty) * cell_competency * tetramer_fraction, 3)

def get_transcriptional_restoration(evaluation, cell_line, tetramer_fraction, post_correction=False):
    results = {}
    dms = 1.0 if post_correction else evaluation.dms_score
    for target in domain_penalties:
        domain_penalty = 0.0 if post_correction else get_domain_penalty(target, evaluation.domain)
        cell_comp = cell_line_competency[cell_line][target]
        score = score_target(dms, domain_penalty, cell_comp, tetramer_fraction)
        results[target] = score
    return results