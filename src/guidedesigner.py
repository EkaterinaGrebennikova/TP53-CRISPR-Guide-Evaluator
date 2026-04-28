from mutationparser import NucleotideChange, get_reference
from utils import complement

spacer_length = 20
editing_window = range(3, 10)  # positions 3-9, 1-based from PAM-distal end
PAM = 'NGG'
pbs_length = 13
rtt_extra = 5

cas9_PAMs = {'SpCas9': ['NGG'], 'SaCas9': ['NNGRRT'], 'Cas9-NG': ['NG'], 'SpRY': ['NRN', 'NYN']}

def pam_matches(seq, pattern):
    if len(seq)< len(pattern):
        return False
    for i, s in enumerate(pattern):
        if s == 'N':
            continue
        if s == 'R':
            if seq[i] == 'A' or seq[i] == 'G':
                continue
            else:
                return False
        if s == 'Y':
            if seq[i] == 'C' or seq[i] == 'T':
                continue
            else:
                return False
        if s == 'G' and seq[i] != 'G':
            return False
        if s == 'C' and seq[i] != 'C':
            return False
        if s == 'A' and seq[i] !='A':
            return False
        if s == 'T' and seq[i] != 'T':
            return False
    return True

def design_guide_cbe_abe(nt_change: NucleotideChange, cds_sequence: str) -> list:
    results = []
    target = nt_change.cds_pos
    for pos in editing_window:
        spacer_start = target-(pos-1)
        if spacer_start < 0 or spacer_start + 22 >= len(cds_sequence):
            continue
        seq = cds_sequence[spacer_start : spacer_start+20]
        for variant, pam_list in cas9_PAMs.items():
            for pam in pam_list:
                pam_seq = cds_sequence[spacer_start+20: spacer_start+20+len(pam)]
                if pam_matches(pam_seq, pam):
                    results.append({"spacer": seq, 'target_pos_in_spacer': pos, 'strand': 'sense', 'cas9_variant': variant, 'pam': pam_seq})
        rc_spacer_end = target + pos
        rc_spacer_start = rc_spacer_end - spacer_length
        if rc_spacer_start >= 0 and rc_spacer_end <= len(cds_sequence):
            for variant, pam_list in cas9_PAMs.items():
                for pam in pam_list:
                    pam_start = rc_spacer_start - len(pam)
                    if pam_start < 0:
                        continue
                    pam_seq = ''.join(complement(c) for c in cds_sequence[pam_start : pam_start+len(pam)])[::-1]
                    if pam_matches(pam_seq, pam):
                        raw = cds_sequence[rc_spacer_start : rc_spacer_end]
                        spacer = ''.join(complement(n) for n in reversed(raw))
                        results.append({"spacer": spacer, "target_pos_in_spacer": pos, "strand": "antisense", 'cas9_variant': variant, 'pam': pam_seq})

    return results

def design_guide_pe(nt_change: NucleotideChange, cds_sequence):
    results = []
    target = nt_change.cds_pos
    for spacer_start in range(target-40, target):
        if spacer_start < 0 or spacer_start + 17 < pbs_length:
            continue
        for variant, pam_list in cas9_PAMs.items():
            for pam in pam_list:
                pam_seq = cds_sequence[spacer_start+20 : spacer_start+20+len(pam)]
                if pam_matches(pam_seq, pam):
                    nick_pos = spacer_start+17
                    rtt_end = target+rtt_extra
                    if rtt_end > len(cds_sequence) or target - nick_pos < 0:
                        continue
                    rtt_raw = list(cds_sequence[nick_pos:rtt_end])
                    rtt_raw[target-nick_pos] = nt_change.ref_nt
                    rtt = ''.join(rtt_raw)
                    pbs_raw = cds_sequence[nick_pos - pbs_length : nick_pos]
                    pbs = ''.join(complement(n) for n in reversed(pbs_raw))
                    results.append({
                        "spacer": cds_sequence[spacer_start : spacer_start+20],
                        "pbs": pbs,
                        "rtt": rtt,
                        "strand": "sense",
                        "nick_to_edit_dist": target - nick_pos,
                        "cas9_variant": variant,
                        'pam': pam_seq
                    })
    return results

def design_guide(nt_change: NucleotideChange, modality, cds_sequence):
    if modality in ('CBE', 'ABE'):
        guides = design_guide_cbe_abe(nt_change, cds_sequence)
        if len(guides) == 0:
            modality = 'Prime Editing'
        else:
            return guides
    if modality == 'Prime Editing':
        return design_guide_pe(nt_change, cds_sequence)
    if modality == 'HDR':
        return [{"spacer": None, "notes": "HDR not yet implemented"}]
    
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from mutationparser import parse_mutations, get_reference
    from modalityselector import select_modalities

    ref = get_reference()
    mutations = parse_mutations(["R175H", "R248W"])
    for m in mutations:
        for item in select_modalities(m):
            modality  = item["modality"]
            nt_change = item["nt_change"]
            guides = design_guide(nt_change, modality, ref.cds_sequence)
            print(f"\n{m.ref_aa}{m.aa_position}{m.alt_aa} | {modality}")
            for g in guides:
                if g.get("spacer") and "pbs" in g:
                    print(f"  spacer: {g['spacer']}  pbs: {g['pbs']}  rtt: {g['rtt']}  strand: {g['strand']}")
                elif g.get("spacer"):
                    print(f"  spacer: {g['spacer']}  window_pos: {g['target_pos_in_spacer']}  strand: {g['strand']}  cas9: {g.get('cas9_variant','?')}")
                else:
                    print(f"  {g.get('notes')}")

