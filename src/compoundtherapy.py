compounds = {
    'Nutlin-3': {
        'mechanism': 'MDM2 inhibitor -- prevents MDM2-mediated p53 degradation',
        'use_when':  'MDM2 amplified or correction efficiency borderline'
    },
    'APR-246': {
        'mechanism': 'Refolds mutant p53 into wild-type conformation',
        'use_when':  'Aggregating mutants (R175H, R248W, R249S)'
    },
    'PRIMA-1': {
        'mechanism': 'Reactivates mutant p53 transcriptional activity',
        'use_when':  'GOF mutations with partial function'
    },
}

def get_compound_synergy(aa_change, aggregates, passes_threshold, mdm2_status, is_gof):
    recommendations = []
    if mdm2_status == 'Amplified' or passes_threshold == False:
        recommendations.append({'name': 'Nutlin-3', 'mechanism': compounds['Nutlin-3']['mechanism'], 'reason': 'MDM2 amplified or correction efficiency borderline, prevents MDM2-based degradation of p53'})
    if aggregates == True:
         recommendations.append({'name': 'APR-246', 'mechanism': compounds['APR-246']['mechanism'], 'reason': 'Aggregating mutants present - difficult to correct with CRISPR. Refolds aggregating mutants into wild-type conforming'})
    if is_gof and passes_threshold == False:
        recommendations.append({'name': 'PRIMA-1', 'mechanism': compounds['PRIMA-1']['mechanism'], 'reason': 'GOF mutation with insufficient correction -- PRIMA-1 may reactivate residual mutant p53'})
    return recommendations