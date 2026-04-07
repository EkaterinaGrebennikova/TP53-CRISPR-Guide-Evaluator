aggregating_mutants = ['R175H', 'R248W', 'R249S', 'R282W', 'G245S']

def get_aggregation_risk(aa_change):
    if aa_change in aggregating_mutants:
        return {
            'aggregates': True,
            'note':  'Mutant protein forms amyloid-like aggregates, correction may not dissolve existing aggregates'
        }
    return {
        'aggregates': False,
        'note': None
    }