cell_line_division =  {
    'HCT116': {'rate': 'Fast',   'hdr_note': None},
    'U2OS':   {'rate': 'Fast',   'hdr_note': None},
    'MCF7':   {'rate': 'Slow',   'hdr_note': 'MCF7 divides slowly — HDR efficiency reduced, consider PE instead'},
}
def get_cell_cycle(modality, cell_line):
    if modality == 'HDR':
        info = cell_line_division.get(cell_line,{})
        return info.get('hdr_note', None)
    return None