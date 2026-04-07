from utils import codon_table, translate_codon
from dmsscorer import get_dms_score

def get_bystander_consequence(guide, modality, nt_change, cds_sequence):
    spacer = guide['spacer']
    target_pos_in_spacer = guide['target_pos_in_spacer']
    results = []
    if modality == 'Prime Editing':
        return []
    for i in range(3, 10):  
        if i == target_pos_in_spacer:
            continue
        if modality == 'CBE' and spacer[i-1] == 'C':
            cds_pos = nt_change.cds_pos + (i-target_pos_in_spacer)
            codon_number = cds_pos // 3
            pos_in_codon = cds_pos % 3
            codon_start = codon_number * 3
            og_codon = cds_sequence[codon_start : codon_start+3]
            mutant_codon = list(og_codon)
            mutant_codon[pos_in_codon] = 'T'
            mutant_codon = ''.join(mutant_codon)
            og_aa = translate_codon(og_codon)
            new_aa = translate_codon(mutant_codon)
            if og_aa != new_aa:
                aa_change = f"{og_aa}{codon_number + 1}{new_aa}"
                dms_score = get_dms_score(aa_change)
                if dms_score is None:
                    consequence = 'unknown'
                elif dms_score > 0.75:
                    consequence = 'benign'
                elif dms_score > 0.25:
                    consequence = 'partial loss'
                else:
                    consequence = 'pathogenic'
                results.append({
                    "spacer_pos": i,
                    "cds_pos": cds_pos,
                    "aa_change": aa_change,
                    "dms_score": dms_score,
                    "consequence": consequence
                })
        elif modality == 'ABE' and spacer[i-1] == 'A':
            cds_pos = nt_change.cds_pos + (i-target_pos_in_spacer)
            codon_number = cds_pos // 3
            pos_in_codon = cds_pos % 3
            codon_start = codon_number * 3
            og_codon = cds_sequence[codon_start : codon_start+3]
            mutant_codon = list(og_codon)
            mutant_codon[pos_in_codon] = 'G'
            mutant_codon = ''.join(mutant_codon)
            og_aa = translate_codon(og_codon)
            new_aa = translate_codon(mutant_codon)
            if og_aa != new_aa:
                aa_change = f"{og_aa}{codon_number + 1}{new_aa}"
                dms_score = get_dms_score(aa_change)
                if dms_score is None:
                    consequence = 'unknown'
                elif dms_score > 0.75:
                    consequence = 'benign'
                elif dms_score > 0.25:
                    consequence = 'partial loss'
                else:
                    consequence = 'pathogenic'
                results.append({
                    "spacer_pos": i,
                    "cds_pos": cds_pos,
                    "aa_change": aa_change,
                    "dms_score": dms_score,
                    "consequence": consequence
                })
    return results

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from mutationparser import parse_mutations, get_reference
    from modalityselector import select_modalities
    from guidedesigner import design_guide

    ref = get_reference()
    for m in parse_mutations(["R248W"]):
        for item in select_modalities(m):
            guides = design_guide(item['nt_change'], item['modality'], ref.cds_sequence)
            for g in guides[:2]:
                if g.get('spacer') and 'target_pos_in_spacer' in g:
                    bystanders = get_bystander_consequence(g, item['modality'], item['nt_change'], ref.cds_sequence)
                    print(f"\nGuide: {g['spacer']}")
                    for b in bystanders:
                        print(f"  {b['aa_change']}  consequence: {b['consequence']}  dms: {b['dms_score']}")
