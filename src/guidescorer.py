from guidedesigner import editing_window
from bystanderevaluator import get_bystander_consequence

CAS9_EFFICIENCY = {
    'SpCas9':  1.0,
    'SaCas9':  0.85,
    'Cas9-NG': 0.80,
    'SpRY':    0.70
}

def score_gc(spacer):
    total=0
    for i in range(len(spacer)):
        if spacer[i] == 'G' or spacer[i] == 'C':
            total+=1
    total = total/len(spacer)
    if 0.4 <= total <= 0.6:
        return 1.0
    elif total < 0.4:
        return max(0.0, 1.0 - (0.4 - total) * 5)
    else:
        return max(0.0, 1.0 - (total - 0.6) * 5)
    
def score_bystander(bystander_consequences:list):
    if len(bystander_consequences) == 0:
        return 1.0
    harms = []
    for b in bystander_consequences:
        if b['dms_score'] is None:
            harms.append(0.3)
        else:
            harms.append(1.0-b['dms_score'])
    avg_harm = sum(harms) / len(harms)
    count_penalty = min(len(bystander_consequences) * 0.05, 0.2)
    return max(0.0, round(1.0 - avg_harm - count_penalty, 3))  

def score_guide(guide, modality, nt_change=None, cds_sequence=None):
    if guide.get("spacer") is None:
        return 0.0
    spacer = guide["spacer"]
    gc = score_gc(spacer)
    variant = guide.get('cas9_variant', 'SpCas9')
    efficiency = CAS9_EFFICIENCY.get(variant, 0.7)
    if "target_pos_in_spacer" in guide and nt_change and cds_sequence:
        bystanders = get_bystander_consequence(guide, modality, nt_change, cds_sequence)
        bystander = score_bystander(bystanders)
        return round((gc * 0.5 + bystander * 0.5) * efficiency, 3)
    else:
        return round(gc * efficiency, 3)

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from mutationparser import parse_mutations, get_reference
    from modalityselector import select_modalities
    from guidedesigner import design_guide

    ref = get_reference()
    mutations = parse_mutations(["R175H", "R248W"])
    for m in mutations:
        for item in select_modalities(m):
            modality  = item["modality"]
            nt_change = item["nt_change"]
            guides = design_guide(nt_change, modality, ref.cds_sequence)
            print(f"\n{m.ref_aa}{m.aa_position}{m.alt_aa} | {modality}")
            for g in guides:
                score = score_guide(g, modality)
                spacer = g.get("spacer") or "N/A"
                print(f"  score: {score}  spacer: {spacer}")