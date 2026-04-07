def score_rna_offtarget(spacer, modality):
    total = 0
    if modality == 'PE' or modality == 'HDR':
        return 1.0
    elif modality == 'ABE':
        for i in range(len(spacer)-1):
            if spacer[i] == 'A' and spacer[i+1] == 'C':
                total += 1
    else:
        for i in range(len(spacer)-1):
            if spacer[i] == 'T' and spacer[i+1] == 'C':
                total += 1
    return max(0.0, 1.0 - 0.15*total)