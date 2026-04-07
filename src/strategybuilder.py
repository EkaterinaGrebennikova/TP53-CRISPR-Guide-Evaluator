from mutationparser import ParsedMutation, get_reference
from mutationevaluator import MutationEvaluation
from modalityselector import select_modalities
from guidedesigner import design_guide
from guidescorer import score_guide
from offtargetscorer import score_offtarget

def build_strategy(parsed_mutation: ParsedMutation, evaluation: MutationEvaluation):
    ref = get_reference()
    cds_sequence = ref.cds_sequence
    nt_mod = select_modalities(parsed_mutation)
    prev_candidate_score = -1.0
    bestguide = None
    best_modality = None
    all_candidates = []
    for item in nt_mod:
        candidate = design_guide(item['nt_change'], item['modality'], cds_sequence)
        for guide in candidate:
            s = score_guide(guide, item['modality'], item['nt_change'], cds_sequence) * 0.6 + score_offtarget(guide['spacer']) * 0.4 if guide.get('spacer') else 0.0
            modality_label = 'Prime Editing' if 'pbs' in guide else item['modality']
            if guide.get('spacer'):
                all_candidates.append({
                    "guide": guide,
                    "modality": modality_label,
                    "score": round(s, 3)
                })
            if s > prev_candidate_score:
                prev_candidate_score = s
                bestguide = guide
                best_modality = modality_label
    return {
        "mutation": parsed_mutation,
        "evaluation": evaluation,
        "modality": best_modality,
        "best_guide": bestguide,
        "score": prev_candidate_score,
        "all_guides": all_candidates
    }

def build_strategies(parsed_mutations, evaluations):
    results = []
    for mutation, evaluation in zip(parsed_mutations, evaluations):
        results.append(build_strategy(mutation, evaluation))
    return results

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from mutationparser import parse_mutations
    from mutationevaluator import evaluate_mutations

    mutations   = parse_mutations(["R175H", "R248W"])
    evaluations = evaluate_mutations(mutations)
    strategies  = build_strategies(mutations, evaluations)

    for s in strategies:
        m = s["mutation"]
        print(f"\n{m.ref_aa}{m.aa_position}{m.alt_aa}")
        print(f"  Modality:  {s['modality']}")
        print(f"  Score:     {s['score']}")
        print(f"  Spacer:    {s['best_guide'].get('spacer') if s['best_guide'] else 'None'}")


