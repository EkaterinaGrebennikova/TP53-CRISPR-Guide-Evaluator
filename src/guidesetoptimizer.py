from utils import complement

def guides_overlap(strategy1, strategy2):
    s1_pos = strategy1['mutation'].nt_changes[0].genomic_pos
    s1_chrom = strategy1['mutation'].nt_changes[0].chrom
    s2_pos = strategy2['mutation'].nt_changes[0].genomic_pos
    s2_chrom = strategy2['mutation'].nt_changes[0].chrom
    if s1_chrom == s2_chrom and abs(s1_pos-s2_pos) <= 20:
        return True
    return False

def guides_complementary(spacer1, spacer2):
    rc = ''.join(complement(n) for n in reversed(spacer2))
    for i in range(len(spacer1) - 7):
        chunk = spacer1[i:i+8]
        if chunk in rc:
            return True
    return False

def optimize_guide_set(strategies):
    conflicts = []
    conflict_pairs = set()
    modalities = set(s['modality'] for s in strategies)

    def mut_name(s):
        return f"{s['mutation'].ref_aa}{s['mutation'].aa_position}{s['mutation'].alt_aa}"

    sorted_strategies = sorted(strategies, key=lambda x: x['score'], reverse=True)

    for i in range(len(strategies)):
        for j in range(i+1, len(strategies)):
            s1 = strategies[i]
            s2 = strategies[j]
            spacer1 = s1['best_guide'].get('spacer') if s1['best_guide'] else None
            spacer2 = s2['best_guide'].get('spacer') if s2['best_guide'] else None
            name1 = mut_name(s1)
            name2 = mut_name(s2)

            if guides_overlap(s1, s2):
                conflicts.append(f"{name1} and {name2}: target sites within 20bp - guides may compete")
                conflict_pairs.add((name1, name2))

            if spacer1 and spacer2 and guides_complementary(spacer1, spacer2):
                conflicts.append(f"{name1} and {name2}: spacers are complementary - guide-guide binding risk")
                conflict_pairs.add((name1, name2))

    # greedy grouping — put non-conflicting guides together
    assigned = set()
    delivery_groups = []
    for s in sorted_strategies:
        name = mut_name(s)
        if name in assigned:
            continue
        group = [name]
        assigned.add(name)
        for other_s in sorted_strategies:
            other = mut_name(other_s)
            if other in assigned:
                continue
            conflicts_with_group = any(
                (g, other) in conflict_pairs or (other, g) in conflict_pairs
                for g in group
            )
            if not conflicts_with_group:
                group.append(other)
                assigned.add(other)
        delivery_groups.append(group)

    return {
        'compatible':      len(conflict_pairs) == 0,
        'conflicts':       conflicts,
        'co_delivery':     len(modalities) == 1 and len(conflict_pairs) == 0,
        'modalities':      list(modalities),
        'delivery_groups': delivery_groups,
    }