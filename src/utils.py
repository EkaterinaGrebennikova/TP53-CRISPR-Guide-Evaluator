#TP53 reference info/indentifiers
TP53_ensembl_transcript = 'ENST00000269305'
TP53_refseq_id = 'NM_000546.6'
TP53_cds_length = 1182
TP53_codon_num = 393
TP53_chrom = 'chr17'
TP53_strand = -1

codon_table = {'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
                'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
                'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
                'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
                'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
                'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
                'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
                'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
                'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
                'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
                'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
                'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
                'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
                'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
                'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
                'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'}

aa_to_codon = {}
for key, value in codon_table.items():
    if value not in aa_to_codon:
        aa_to_codon[value] = [key]
    else:
        aa_to_codon[value].append(key)

aa_one_three = {'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp',
                'C': 'Cys', 'E': 'Glu', 'Q': 'Gln', 'G': 'Gly',
                'H': 'His', 'I': 'Ile', 'L': 'Leu', 'K': 'Lys',
                'M': 'Met', 'F': 'Phe', 'P': 'Pro', 'S': 'Ser',
                'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val', '*': 'Stop'}

def translate_codon(codon):
    if codon in codon_table:
        return codon_table[codon]
    else:
        return 'Codon entered incorrectly'

def complement(n):
    if n == 'T':
        return 'A'
    if n == 'A':
        return 'T'
    if n == 'C':
        return 'G'
    if n == 'G':
        return 'C'
    
def find_single_nt_paths(ref_codon, alt_aa):
    paths = []
    for alt_codon in aa_to_codon.get(alt_aa, []):
        diffs = []
        for i in range(3):
            if ref_codon[i]!=alt_codon[i]:
                diffs.append((i, ref_codon[i], alt_codon[i]))
        if len(diffs)==1:
            pos, ref_nt, alt_nt = diffs[0]
            paths.append({'codon_pos': pos, 'ref_nt': ref_nt, 'alt_nt': alt_nt, 'alt_codon': alt_codon})
    return paths
