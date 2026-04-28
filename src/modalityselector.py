from mutationparser import ParsedMutation, NucleotideChange

modality_CBE = "CBE"
modality_ABE = 'ABE'
modality_PE = 'Prime Editing'
modality_HDR = 'HDR'

def select_modality(nt_change: NucleotideChange):
    if nt_change is None:
        return modality_HDR
    elif nt_change.ref_nt == 'C' and nt_change.alt_nt == 'T':
        return modality_ABE
    elif nt_change.ref_nt == 'G' and nt_change.alt_nt == 'A':
        return modality_ABE
    elif nt_change.ref_nt == 'A' and nt_change.alt_nt == 'G':
        return modality_CBE
    elif nt_change.ref_nt == 'T' and nt_change.alt_nt == 'C':
        return modality_CBE
    else:
        return modality_PE
    
def select_modalities(parsed_mutation: ParsedMutation):
    l = []
    if len(parsed_mutation.nt_changes) == 0:
        return [{"nt_change": None, "modality": modality_HDR}]
    for nt_change in parsed_mutation.nt_changes:
        l.append({"nt_change": nt_change, "modality": select_modality(nt_change)})
    return l

