from tcgaloader import load_mutations, load_cna, load_clinical, merge_patient_data

def get_mutation_frequencies(df):
    counts = df['aa_change'].value_counts()
    total = len(df)
    result = []
    for aa_change, count in counts.items():
        result.append({
            'aa_change': aa_change,
            'count': int(count),
            'fraction': round(count / total, 3)
        })
    return result

def get_frequencies_by_cancer_type(df):
    results = {}
    for cancer_type, group_df in df.groupby('cancer_type'):
        counts = group_df['aa_change'].value_counts()
        total = len(group_df)
        aa_list = []
        for aa_change, count in counts.items():
            aa_list.append({
                'aa_change': aa_change,
                'count': int(count),
                'fraction': round(count / total, 3)
            })
        results[cancer_type] = aa_list
    return results

def get_correctable_fraction(df):
    total = len(df['Reference_Allele'])
    cbe_count = 0
    abe_count = 0
    pe_count = 0
    hdr_count = 0
    for ref, alt in zip(df['Reference_Allele'], df['Tumor_Seq_Allele2']):
        if (ref == 'C' and alt == 'T') or (ref == 'G' and alt == 'A'):
           abe_count += 1
        elif (ref == 'T' and alt == 'C') or (ref == 'A' and alt == 'G'):
           cbe_count += 1
        elif len(ref) == 1 and len(alt) == 1 and ref in 'ACGT' and alt in 'ACGT':
            pe_count += 1
        else:
            hdr_count += 1
    return {
        'total': total,
        'cbe': {
            'count': cbe_count,
            'fraction': round(cbe_count/total, 3)
        },
        'abe': {
            'count': abe_count,
            'fraction': round(abe_count/total, 3)
        },
        'pe': {
            'count': pe_count,
            'fraction': round(pe_count/total, 3)
        },
        'hdr': {
            'count': hdr_count,
            'fraction': round(hdr_count/total, 3)
        },
        'base_editable': {
            'count': cbe_count + abe_count,
            'fraction': round((cbe_count + abe_count)/total, 3)
        }
    }

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    muts = load_mutations()
    cna = load_cna()
    clin = load_clinical()
    merged = merge_patient_data(muts, cna, clin)
    print(f"Total TP53 mutations: {len(merged)}")
    print(f"\nTop 10 mutations:")
    for entry in get_mutation_frequencies(merged)[:10]:
        print(f"  {entry['aa_change']}: {entry['count']} ({entry['fraction']*100:.1f}%)")
    print(f"\nCorrectable fraction:")
    result = get_correctable_fraction(merged)
    for key in ['cbe', 'abe', 'pe', 'hdr', 'base_editable']:
        print(f"  {key}: {result[key]['count']} ({result[key]['fraction']*100:.1f}%)")
