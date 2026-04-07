import re, os, json
from dataclasses import dataclass, field
from typing import List, Optional
import requests
from Bio import Entrez, SeqIO
from utils import (
    TP53_ensembl_transcript, TP53_refseq_id, TP53_cds_length,
    TP53_codon_num, TP53_chrom, TP53_strand,
    translate_codon, complement, find_single_nt_paths
)

Entrez.email = "catherine.grebennikova@gmail.com"

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_FILE = os.path.join(DATA_DIR, 'tp53_reference_cache.json')

@dataclass
class NucleotideChange:
    codon_pos: int   # 0-based position within the codon (0, 1, or 2)
    ref_nt: str   # reference nucleotide (on CDS/coding strand)
    alt_nt: str   # alternate nucleotide (on CDS/coding strand)
    alt_codon: str   # resulting codon after the change
    cds_pos: int   # 0-based position in the CDS
    chrom: str   # e.g. "chr17"
    genomic_pos: int   # 1-based hg38 genomic position
    genomic_ref_nt: str   # ref nucleotide on the FORWARD genomic strand
    genomic_alt_nt: str   # alt nucleotide on the FORWARD genomic strand

@dataclass
class ParsedMutation:
    raw_input: str
    ref_aa: str
    aa_position: int
    alt_aa: str
    ref_codon: str
    cds_codon_start: int                              
    nt_changes: List[NucleotideChange] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    def summary(self):
        return(f"{self.ref_aa}{self.aa_position}{self.alt_aa} | "
               f"codon: {self.ref_codon} | "
               f"CDS position: {self.cds_codon_start} | "
               f"{len(self.nt_changes)} nucleotide path(s)")

class TP53Reference:
    """
    Holds two things:
      1. The TP53 CDS sequence (1182 nucleotides, from NCBI)
      2. A map from every CDS position → its genomic coordinate on hg38 (from Ensembl)
    Fetches both on first run and caches to data/tp53_reference_cache.json.
    """

    def __init__(self):
        self.cds_sequence  = ""    # the 1182-nt CDS string
        self.cds_to_genome = {}    # {cds_pos (int): {"chrom", "pos", "strand"}}
        self._load()

    def _load(self):
        """Load from cache if it exists, otherwise fetch fresh and save."""
        if os.path.exists(CACHE_FILE):
            print("[Reference] Loading cached TP53 reference data...")
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            self.cds_sequence  = cache["cds_sequence"]
            # JSON keys are always strings — convert back to int
            self.cds_to_genome = {int(k): v for k, v in cache["cds_to_genome"].items()}
            print(f"[Reference] Loaded {len(self.cds_sequence)} nt CDS, "
                  f"{len(self.cds_to_genome)} positions mapped.")
        else:
            print("[Reference] No cache found. Fetching from NCBI + Ensembl...")
            self._fetch()
            self._save()

    def _fetch(self):
        self.cds_sequence  = self._fetch_cds_ncbi()
        self.cds_to_genome = self._fetch_coord_map_ensembl()

    def _fetch_cds_ncbi(self) -> str:
        """
        Fetch NM_000546.6 from NCBI in GenBank format.
        Find the CDS feature and extract its nucleotide sequence.
        """
        print(f"[NCBI] Fetching {TP53_refseq_id}...")
        handle = Entrez.efetch(
            db="nucleotide",
            id=TP53_refseq_id,
            rettype="gb",
            retmode="text"
        )
        record = SeqIO.read(handle, "genbank")
        handle.close()

        for feature in record.features:
            if feature.type == "CDS":
                cds_seq = str(feature.extract(record.seq))
                print(f"[NCBI] CDS extracted: {len(cds_seq)} nt")
                if len(cds_seq) != TP53_cds_length:
                    raise ValueError(
                        f"CDS length {len(cds_seq)} != expected {TP53_cds_length}."
                    )
                return cds_seq

        raise ValueError("No CDS feature found in the NCBI record.")

    def _fetch_coord_map_ensembl(self) -> dict:
        """
        Use the Ensembl REST API to map every CDS position to a genomic coordinate.

        The endpoint /map/cds/{transcript}/{start}..{end} returns a list of
        mappings — one per exon. Each mapping covers a chunk of CDS positions
        and tells us the corresponding genomic coordinates.

        For the minus strand (TP53):
          CDS position 0 → highest genomic coordinate in that exon
          CDS position 1 → one lower, etc.
        """
        print(f"[Ensembl] Fetching coordinate map for {TP53_ensembl_transcript}...")
        url = (f"https://rest.ensembl.org/map/cds/"
               f"{TP53_ensembl_transcript}/1..{TP53_cds_length}")
        headers = {"Content-Type": "application/json"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        mappings = response.json()["mappings"]
        cds_to_genome = {}
        cds_pos = 0  # accumulated 0-based CDS position across all exons

        for mapping in mappings:
            # Ensembl returns genomic coords directly on the mapping object
            gen_start = mapping["start"]               # lower genomic coordinate, 1-based
            gen_end   = mapping["end"]                 # higher genomic coordinate, 1-based
            strand    = mapping["strand"]              # 1 or -1
            chrom     = "chr" + str(mapping["seq_region_name"])

            seg_len = gen_end - gen_start + 1         # number of nucleotides in this exon

            for i in range(seg_len):
                if strand == -1:
                    # minus strand: first CDS position maps to highest genomic coord
                    genomic_pos = gen_end - i
                else:
                    genomic_pos = gen_start + i

                cds_to_genome[cds_pos] = {
                    "chrom":  chrom,
                    "pos":    genomic_pos,             # 1-based
                    "strand": strand
                }
                cds_pos += 1

        print(f"[Ensembl] Mapped {len(cds_to_genome)} CDS positions "
              f"across {len(mappings)} exons.")
        return cds_to_genome

    def _save(self):
        """Save fetched data to the cache file."""
        os.makedirs(DATA_DIR, exist_ok=True)
        cache = {
            "cds_sequence":  self.cds_sequence,
            # JSON requires string keys
            "cds_to_genome": {str(k): v for k, v in self.cds_to_genome.items()},
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"[Reference] Saved cache to {CACHE_FILE}")

    def get_codon(self, aa_position: int) -> str:
        start = (aa_position-1)*3
        return self.cds_sequence[start : start+3]

    def get_genomic_coords(self, cds_pos: int) -> dict:
        return self.cds_to_genome[cds_pos]        

    def genomic_to_cds(self, chrom: str, genomic_pos: int) -> Optional[int]:
        for cds_pos, i in self.cds_to_genome.items():
            if i["chrom"]==chrom and i["pos"]==genomic_pos:
                return cds_pos
        return None

shorthand_pattern = re.compile(r'^([A-Z])(\d+)([A-Z])$', re.IGNORECASE)
hgvs_pattern = re.compile(r'^c\.(\d+)([ACGT])>([ACGT])$', re.IGNORECASE) 

def parse_shorthand(s):
    match = shorthand_pattern.match(s)
    if match:
        ref_aa, pos, alt_aa = match.groups()
        return {'ref_aa': ref_aa.upper(), "aa_pos": int(pos), 'alt_aa': alt_aa.upper()}
    return None

def parse_hgvs(s):
    match = hgvs_pattern.match(s)
    if match:
        cds_pos, ref_nt, alt_nt = match.groups()
        return {'cds_pos': int(cds_pos), 'ref_nt': ref_nt.upper(), 'alt_nt': alt_nt.upper()}
    return None

def parse_vcf(s):
    parts = s.strip().split()
    if len(parts)<5:
        return None
    
    chrom = parts[0]
    pos = parts[1]
    ref = parts[3]
    alt = parts[4]

    if not chrom.startswith("chr"):
        chrom = "chr"+chrom

    if chrom != "chr17":
        return None
    
    return{'chrom': chrom, 'pos': int(pos), 'ref': ref.upper(), 'alt': alt.upper()}

def map_shorthand(parsed, ref: TP53Reference):
    ref_aa = parsed['ref_aa']
    aa_pos = parsed['aa_pos']
    alt_aa = parsed['alt_aa']
    cds_codon_start = (aa_pos - 1) * 3
    ref_codon = ref.get_codon(aa_pos)
    raw_input = f"{ref_aa}{aa_pos}{alt_aa}"

    if not (1 <= aa_pos <= 393):
        raise ValueError(f"Position {aa_pos} out of range (1-393)")

    if translate_codon(ref_codon) != ref_aa:
        raise ValueError(
            f"Mismatch at position {aa_pos}: "
            f"CDS has '{translate_codon(ref_codon)}' but input says '{ref_aa}'"
        )

    paths = find_single_nt_paths(ref_codon, alt_aa)
    mutation = ParsedMutation(
        raw_input = raw_input,
        ref_aa = ref_aa,
        aa_position = aa_pos,
        alt_aa = alt_aa,
        ref_codon = ref_codon,
        cds_codon_start = cds_codon_start
    )

    for path in paths:
        cds_pos = cds_codon_start + path['codon_pos']
        coords = ref.get_genomic_coords(cds_pos)
        nc = NucleotideChange(
            codon_pos = path['codon_pos'],
            ref_nt = path['ref_nt'],
            alt_nt = path['alt_nt'],
            alt_codon = path['alt_codon'],
            cds_pos = cds_pos,
            chrom = coords['chrom'],
            genomic_pos = coords['pos'],
            genomic_ref_nt = complement(path['ref_nt']),
            genomic_alt_nt = complement(path['alt_nt'])
        )
        mutation.nt_changes.append(nc)

    return mutation

def map_hgvs(parsed, ref : TP53Reference):
    cds_pos_1_based = parsed['cds_pos']
    cds_pos = cds_pos_1_based-1

    if not (0 <= cds_pos <= 1181):
        raise ValueError(f"Position {cds_pos_1_based} out of range (1-1182)")

    ref_nt = parsed['ref_nt']
    alt_nt = parsed['alt_nt']

    if ref.cds_sequence[cds_pos] != ref_nt:
        raise ValueError(
            f"Mismatch at position {cds_pos_1_based}: "
            f"CDS has '{ref.cds_sequence[cds_pos]}' but input says '{ref_nt}'"
        )

    codon_index = cds_pos // 3
    aa_pos = codon_index + 1
    codon_pos = cds_pos % 3
    ref_codon = ref.get_codon(aa_pos)
    alt_codon_list = list(ref_codon)
    alt_codon_list[codon_pos] = alt_nt
    alt_codon = ''.join(alt_codon_list)
    raw_input = f"c.{cds_pos_1_based}{ref_nt}>{alt_nt}"
    ref_aa = translate_codon(ref_codon)
    alt_aa = translate_codon(alt_codon)

    if ref_aa == alt_aa:
        raise ValueError(f"c.{cds_pos_1_based}{ref_nt}>{alt_nt} is synonymous — no amino acid change")

    coords = ref.get_genomic_coords(cds_pos)
    mutation = ParsedMutation(
        raw_input = raw_input,
        ref_aa = ref_aa,
        aa_position = aa_pos,
        alt_aa = alt_aa,
        ref_codon = ref_codon,
        cds_codon_start = codon_index*3,
        nt_changes = []
    )
    nc = NucleotideChange(
        codon_pos = codon_pos,
        ref_nt = ref_nt,
        alt_nt = alt_nt,
        alt_codon = alt_codon,
        cds_pos = cds_pos,
        chrom = coords["chrom"],
        genomic_pos = coords["pos"],
        genomic_ref_nt = complement(ref_nt),
        genomic_alt_nt = complement(alt_nt)
    )
    mutation.nt_changes.append(nc)
    return mutation
    
def map_vcf(parsed, ref:TP53Reference):
    chrom = parsed['chrom']
    genomic_pos = parsed['pos']
    complement_ref_nt = parsed['ref']
    complement_alt_nt = parsed['alt']
    ref_nt = complement(complement_ref_nt)
    alt_nt = complement(complement_alt_nt)
    cds_pos = ref.genomic_to_cds(chrom, genomic_pos)

    if cds_pos is None:
        raise ValueError(
            f"{chrom}:{genomic_pos} is not in the TP53 CDS — "
            f"check that this position is within a coding exon on hg38"
        )
    
    if ref.cds_sequence[cds_pos] != ref_nt:
        raise ValueError(
            f"Ref mismatch at {chrom}:{genomic_pos}: "
            f"CDS has '{ref.cds_sequence[cds_pos]}' but VCF says '{ref_nt}'"
        )

    codon_index = cds_pos // 3
    aa_pos = codon_index + 1
    codon_pos = cds_pos%3
    cds_codon_start = codon_index*3
    ref_codon = ref.get_codon(aa_pos)
    alt_codon_list = list(ref_codon)
    alt_codon_list[codon_pos] = alt_nt
    alt_codon = ''.join(alt_codon_list)

    ref_aa = translate_codon(ref_codon)
    alt_aa = translate_codon(alt_codon)
    
    if ref_aa == alt_aa:
        raise ValueError(f"{chrom}:{genomic_pos} is synonymous — no amino acid change")
    
    raw_input = f"{chrom}:{genomic_pos}{complement_ref_nt}>{complement_alt_nt}"

    mutation = ParsedMutation(
        raw_input = raw_input,
        ref_aa = ref_aa,
        aa_position = aa_pos,
        alt_aa = alt_aa,
        ref_codon = ref_codon,
        cds_codon_start = cds_codon_start,
        nt_changes = []
    )

    nc = NucleotideChange(
        codon_pos = codon_pos,
        ref_nt = ref_nt,
        alt_nt = alt_nt,
        alt_codon = alt_codon,
        cds_pos = cds_pos,
        chrom = chrom,
        genomic_pos = genomic_pos,
        genomic_ref_nt = complement_ref_nt,
        genomic_alt_nt = complement_alt_nt
    )
    mutation.nt_changes.append(nc)
    return mutation

_reference: Optional[TP53Reference] = None

def get_reference() -> TP53Reference:
    global _reference
    if _reference is None:
        _reference = TP53Reference()
    return _reference

def parse_mutation(s, ref: TP53Reference = None):
    parsed = parse_shorthand(s)
    if parsed:
        return map_shorthand(parsed, ref)
    
    parsed = parse_hgvs(s)
    if parsed:
        return map_hgvs(parsed, ref)
    
    parsed = parse_vcf(s)
    if parsed:
        return map_vcf(parsed, ref)
    
    raise ValueError(f"Unrecognized format: '{s}'")

def parse_mutations(mutation_strings):
    ref = get_reference()
    results = []    
    errors = []
    for s in mutation_strings:
        s = s.strip()
        if not s:
            continue
        try:
            result = parse_mutation(s, ref)
            results.append(result)
            print(f"[Parser] {s} -> {result.summary()}")
        except ValueError as e:
            errors.append((s, str(e)))
            print(f"[Parser] ERROR '{s}': {e}")

    if errors:
        for s, msg in errors:
            print(f"  failed: {s} — {msg}")

    return results
