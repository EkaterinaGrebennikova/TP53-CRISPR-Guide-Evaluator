EFFICIENCY_NOTES = {
    ('CBE',  'HCT116'): ('RNP electroporation', 'Lipofection',
                         '60-80% editing efficiency typical',
                         '30-50% editing efficiency typical'),
    ('CBE',  'U2OS'):   ('RNP electroporation', 'Lipofection',
                         '70-90% editing efficiency typical',
                         '40-60% editing efficiency typical'),
    ('CBE',  'MCF7'):   ('RNP electroporation', 'Lipofection',
                         '40-60% editing efficiency typical',
                         '20-40% editing efficiency typical — low transfectability'),
    ('ABE',  'HCT116'): ('RNP electroporation', 'Lipofection',
                         '60-80% editing efficiency typical',
                         '30-50% editing efficiency typical'),
    ('ABE',  'U2OS'):   ('RNP electroporation', 'Lipofection',
                         '70-90% editing efficiency typical',
                         '40-60% editing efficiency typical'),
    ('ABE',  'MCF7'):   ('RNP electroporation', 'Lipofection',
                         '40-60% editing efficiency typical',
                         '20-40% editing efficiency typical — low transfectability'),
    ('Prime Editing',   'HCT116'): ('Plasmid transfection', 'Lentiviral',
                         '30-50% pegRNA editing efficiency typical',
                         '50-70% editing efficiency typical'),
    ('Prime Editing',   'U2OS'):   ('Plasmid transfection', 'Lentiviral',
                         '40-60% pegRNA editing efficiency typical',
                         '50-70% editing efficiency typical'),
    ('Prime Editing',   'MCF7'):   ('Lentiviral', 'None',
                         '50-70% editing efficiency typical — recommended for low-transfectability lines',
                         'N/A'),
    ('HDR',  'HCT116'): ('RNP + ssODN electroporation', 'AAV donor',
                         '10-30% HDR efficiency typical — HDR less efficient in non-dividing cells',
                         '15-35% HDR efficiency typical with AAV donor template'),
    ('HDR',  'U2OS'):   ('RNP + ssODN electroporation', 'AAV donor',
                         '20-40% HDR efficiency typical — actively dividing cells favor HDR',
                         '15-35% HDR efficiency typical with AAV donor template'),
    ('HDR',  'MCF7'):   ('RNP + ssODN electroporation', 'AAV donor',
                         '5-20% HDR efficiency typical — consider PE as alternative',
                         '15-35% HDR efficiency typical with AAV donor template'),
}

def recommend_delivery(modality, cell_line):
    entry = EFFICIENCY_NOTES.get((modality, cell_line))
    if entry:
        primary, secondary, primary_eff, secondary_eff = entry
        note = 'pegRNA too large for efficient RNP' if modality == 'Prime Editing' else \
               'ssODN delivered alongside RNP'      if modality == 'HDR' else \
               'Minimizes off-target vs plasmid'
    else:
        primary, secondary = 'RNP electroporation', 'Lipofection'
        primary_eff, secondary_eff = 'Efficiency data not available', 'Efficiency data not available'
        note = 'Minimizes off-target vs plasmid'
    return {
        'primary':       primary,
        'secondary':     secondary,
        'note':          note,
        'primary_eff':   primary_eff,
        'secondary_eff': secondary_eff,
    }