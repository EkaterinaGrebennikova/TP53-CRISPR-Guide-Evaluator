# Domain penalties derived from IARC TP53 yeast transactivation assay data
# Penalty = 1 - (mean % WT activity / 100) per domain per target gene
# p21: WAF1nWT (n=1129 DBD, 344 TA1, 198 Tet)
# MDM2: MDM2nWT (same sample sizes)
# BAX: BAXnWT (same sample sizes)
# PUMA: PUMAnWT_Saos2 (n=155 DBD; insufficient TA1/Tet data, using BAX as proxy)
domain_penalties = {
    'p21':  {'DNA-binding': 0.509, 'Transactivation 1': 0.000, 'Tetramerization': 0.186, 'default': 0.046},
    'MDM2': {'DNA-binding': 0.625, 'Transactivation 1': 0.060, 'Tetramerization': 0.000, 'default': 0.184},
    'PUMA': {'DNA-binding': 0.000, 'Transactivation 1': 0.000, 'Tetramerization': 0.107, 'default': 0.081},
    'BAX':  {'DNA-binding': 0.444, 'Transactivation 1': 0.012, 'Tetramerization': 0.107, 'default': 0.081},
}

# Per-target competency from DepMap OmicsCNGeneWGS copy number data
# MDM2 amplification (CN>1.1) suppresses p53→MDM2 axis: comp = 1/CN
# MDM2 hemizygous loss (CN<0.8) enhances p53 signaling: comp = 1.0
# All other genes at diploid CN across these lines → comp = 1.0
cell_line_competency = {
    'HCT116': {'p21': 1.0, 'MDM2': 1.0, 'PUMA': 1.0, 'BAX': 1.0},
    'U2OS':   {'p21': 1.0, 'MDM2': 1.0, 'PUMA': 1.0, 'BAX': 1.0},
    'MCF7':   {'p21': 1.0, 'MDM2': 0.73, 'PUMA': 1.0, 'BAX': 1.0},
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