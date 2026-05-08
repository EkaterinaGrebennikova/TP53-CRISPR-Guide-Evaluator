# comp_score derived from DepMap OmicsCNGeneWGS copy number data
# HCT116: all pathway genes diploid (CN ~1.0) → 1.0
# U2OS: MDM2 CN=0.67 (hemizygous loss helps p53), others diploid → 1.0
# MCF7: MDM2 CN=1.37 (amplified, comp=1/1.37≈0.73), others diploid → mean 0.932
cell_line_status = {
    'HCT116': {'MDM2': 'Wild-type',
                'p21': 'Intact',
                'BAX': 'Intact',
                'PUMA': 'Intact',
                'comp_score': 1.0},
    'U2OS': {'MDM2': 'Hemizygous loss',
             'p21': 'Intact',
             'BAX': 'Intact',
             'PUMA': 'Intact',
             'comp_score': 1.0},
    'MCF7': {'MDM2': 'Amplified',
             'p21': 'Intact',
             'BAX': 'Intact',
             'PUMA': 'Intact',
             'comp_score': 0.932}
    }

def get_pathway_status(cell_line, evaluation, tetramer_fraction, post_correction=False):
    cell_line_info = cell_line_status[cell_line]
    if post_correction:
        structural_recovery = 1.0
    else:
        structural_recovery = 1.0 - (evaluation.structural_impact if evaluation.structural_impact is not None else 0.5)
    prog_score = round(tetramer_fraction * structural_recovery * cell_line_info['comp_score'], 3)
    # Thresholds: favorable ≥ 0.225 (~25% functional tetramers), moderate ≥ 0.113
    if prog_score >= 0.225:
        prognosis = 'Favorable'
    elif prog_score >= 0.113:
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
    