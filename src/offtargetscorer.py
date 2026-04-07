mit_weights = [0.000, 0.000, 0.014, 0.000, 0.000,
                0.395, 0.317, 0.000, 0.389, 0.079,
                0.445, 0.508, 0.613, 0.851, 0.732,
                0.828, 0.615, 0.804, 0.685, 0.583
            ]

seed_length = 12

def score_mit(spacer):
    total = 0
    for i in range(len(spacer)):
        if spacer[i] == 'C' or spacer[i] == 'G':
            total += mit_weights[i]
    return 1.0 - (total/sum(mit_weights))

def score_complexity(spacer):
    total = 1.0
    prev_nc = spacer[0]
    same_count = 1
    for n in spacer[1:]:
        if n == prev_nc:
            same_count += 1
            if same_count == 4:
                total -= 0.25
        else:
            prev_nc = n
            same_count = 1
    gc_count = 0
    for i in range(len(spacer)-12, len(spacer)):
        if spacer[i] == 'G' or spacer[i] == 'C':
            gc_count += 1
    if gc_count/12 > 0.75:
        total -= 0.2

    if (spacer[len(spacer)-1] == 'G' or spacer[len(spacer)-1] == 'C') and (spacer[len(spacer)-2] == 'G' or spacer[len(spacer)-2] == 'C') and (spacer[len(spacer)-3] == 'G' or spacer[len(spacer)-3] == 'C'):
        total -= 0.15
    return max(0.0, total)

def score_offtarget(spacer):
    return round(score_mit(spacer) * 0.6 + score_complexity(spacer) * 0.4, 3)

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from mutationparser import parse_mutations, get_reference
    from modalityselector import select_modalities
    from guidedesigner import design_guide

    ref = get_reference()
    for m in parse_mutations(["R175H", "R248W"]):
        for item in select_modalities(m):
            for g in design_guide(item['nt_change'], item['modality'], ref.cds_sequence):
                if g.get("spacer"):
                    print(f"{m.ref_aa}{m.aa_position}{m.alt_aa}  {g['spacer']}  off-target safety: {score_offtarget(g['spacer'])}")
