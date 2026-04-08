tetramer_threshold = 0.45

def get_allele_status(zygosity, efficiency):
    if zygosity == 'heterozygous':
        pre_wt = 0.5
        effective_wt = round(0.5 + (efficiency * 0.5), 3)
        note = 'One WT allele present; correction adds up to 50% more WT protein'
    elif zygosity == 'homozygous':
        pre_wt = 0.0
        effective_wt = round(efficiency, 3)
        note = 'Both alleles mutant; all WT protein must come from correction'
    elif zygosity == 'loh':
        pre_wt = 0.0
        effective_wt = round(efficiency, 3)
        note = 'WT allele deleted by tumor (LOH at 17p13.1); functionally equivalent to homozygous'
    else:
        raise ValueError(f"Invalid zygosity '{zygosity}': must be 'heterozygous', 'homozygous', or 'loh'")
    pre_tetramer = round(pre_wt ** 4, 3)
    tetramer_fraction = round(effective_wt ** 4, 3)
    return {
        'zygosity': zygosity,
        'pre_correction_wt_fraction': pre_wt,
        'pre_correction_tetramer': pre_tetramer,
        'effective_wt_fraction': effective_wt,
        'tetramer_fraction': tetramer_fraction,
        'passes_threshold': tetramer_fraction >= tetramer_threshold,
        'note': note
    }