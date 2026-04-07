cell_line_status = {
    'HCT116': {'MDM2': 'Wild-type',
                'p21': 'Intact',
                'BAX': 'Intact',
                'PUMA': 'Intact',
                'comp_score': 1.0},
    'U2OS': {'MDM2': 'Wild-type',
             'p21': 'Intact',
             'BAX': 'Intact',
             'PUMA': 'Intact',
             'comp_score': 0.9},
    'MCF7': {'MDM2': 'Amplified',
             'p21': 'Compromised',
             'BAX': 'Intact',
             'PUMA': 'Intact',
             'comp_score': 0.4}
    }

def get_pathway_status(cell_line, evaluation, tetramer_fraction):
    cell_line_info = cell_line_status[cell_line]
    structural_recovery = 1.0 - (evaluation.structural_impact if evaluation.structural_impact is not None else 0.5)
    prog_score = round(tetramer_fraction * structural_recovery * cell_line_info['comp_score'], 3)
    if prog_score >= 0.25:
        prognosis = 'Favorable'
    elif prog_score >= 0.12:
        prognosis = 'Moderate'
    else:
        prognosis = 'Poor'
    return {
        'mdm2':      cell_line_info['MDM2'],
        'p21':       cell_line_info['p21'],
        'bax':       cell_line_info['BAX'],
        'puma':      cell_line_info['PUMA'],
        'score':     prog_score,
        'prognosis': prognosis
    }
    