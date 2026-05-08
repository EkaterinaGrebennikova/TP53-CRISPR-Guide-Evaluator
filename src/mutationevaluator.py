import requests
from dataclasses import dataclass, field
from typing import List
from Bio import Entrez
from dmsscorer import get_dms_score
from structuralscorer import get_structural_impact
from cancerdriverscore import check_driver_overlap
from iarcp53 import get_iarc_annotation

Entrez.email = "catherine.grebennikova@gmail.com"

TP53_Domains = {
    "Transactivation 1": (1, 40),
    "Transactivation 2": (40, 61),
    "Proline-rich": (62, 94),
    "DNA-binding": (94, 292),
    "Linker": (293, 325),
    "Tetramerization": (323, 356),
    "Regulatory": (364, 393)
}

DNA_contact_residues = {120, 241, 248, 273, 276, 277, 280, 281, 283}
GOF_mutations = {
    "R175H", "G245S", "R248W", "R248Q",
    "R249S", "R273H", "R273C", "R282W"
}

@dataclass
class MutationEvaluation:
    aa_position: int
    aa_change: str
    domain: str
    in_dbd: bool
    is_contact_residue: bool
    is_gof: bool
    clinvar_significance: str
    functional_severity: float
    dms_score: float = None
    structural_impact: float = None
    driver_genes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    iarc_transactivation: str = None
    iarc_structure_function: str = None
    iarc_hotspot: bool = False
    iarc_experimental_gof: str = None
    iarc_temperature_sensitive: str = None
    iarc_somatic_count: int = 0
    iarc_germline_count: int = 0

def get_domain(aa_position):
    for key, value in TP53_Domains.items():
        if value[0] <= aa_position <= value[1]:
            return key
    return "Unknown"


# Cache to avoid hitting ClinVar API repeatedly for the same mutation
_clinvar_cache = {}

def get_clinvar_significance(aa_change: str) -> str:
    """
    Query ClinVar via NCBI REST API for the clinical significance of a
    TP53 variant (e.g. "R175H").
    Returns a string like "Pathogenic", "Likely pathogenic", or "Unknown".
    """
    if aa_change in _clinvar_cache:
        return _clinvar_cache[aa_change]

    try:
        # Step 1: search ClinVar for this TP53 variant
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db":      "clinvar",
                "term":    f"TP53[gene] AND {aa_change}",
                "retmode": "json",
                "retmax":  5
            },
            timeout=15
        )
        ids = r.json()["esearchresult"]["idlist"]

        if not ids:
            _clinvar_cache[aa_change] = "Unknown"
            return "Unknown"

        # Step 2: fetch the summary for the first result
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={
                "db":      "clinvar",
                "id":      ids[0],
                "retmode": "json"
            },
            timeout=15
        )
        result_set = r.json().get("result", {})

        # Step 3: extract clinical significance
        doc = result_set.get(ids[0], {})
        sig = doc.get("germline_classification", {})

        if isinstance(sig, dict):
            result = sig.get("description", "Unknown")
        else:
            result = str(sig) if sig else "Unknown"

        _clinvar_cache[aa_change] = result
        return result

    except Exception:
        _clinvar_cache[aa_change] = "Unknown"
        return "Unknown"
    
def compute_severity(in_dbd, is_contact_residue, is_gof, clinvar_significance, dms_score):
    if dms_score is not None:
        return round(1.0 - dms_score, 3)
    else:
        # Weights derived via linear regression on 2,308 IARC-annotated TP53 mutations (R²=0.628)
        score = 0.545  # baseline intercept
        if in_dbd:
            score += 0.120
        if is_contact_residue:
            pass  # coefficient -0.034; contact effect captured by DBD membership
        if is_gof:
            score += 0.066
        if clinvar_significance == 'Pathogenic':
            score += 0.094
        return min(round(score, 3), 1.0)

def evaluate_mutation(parsed_mutation):
    aa_pos = parsed_mutation.aa_position
    aa_change = f"{parsed_mutation.ref_aa}{aa_pos}{parsed_mutation.alt_aa}"
    domain = get_domain(aa_pos)
    in_dbd = domain == 'DNA-binding'
    is_contact_residue = aa_pos in DNA_contact_residues
    iarc = get_iarc_annotation(aa_change)
    if iarc.get('found') and iarc.get('experimental_gof') is not None:
        is_gof = True
    else:
        is_gof = aa_change in GOF_mutations
    clinvar_sig = get_clinvar_significance(aa_change)
    dms_score = get_dms_score(aa_change)
    s_impact = get_structural_impact(aa_pos)
    severity = compute_severity(in_dbd, is_contact_residue, is_gof, clinvar_sig, dms_score)

    genomic_coords = [
        {"chrom": nc.chrom.replace("chr", ""), "pos": nc.genomic_pos}
        for nc in parsed_mutation.nt_changes
    ]
    driver_genes = check_driver_overlap(genomic_coords)

    return MutationEvaluation(
        aa_position = aa_pos,
        aa_change = aa_change,
        domain = domain,
        in_dbd = in_dbd,
        is_contact_residue = is_contact_residue,
        is_gof = is_gof,
        clinvar_significance = clinvar_sig,
        functional_severity = severity,
        dms_score = dms_score,
        structural_impact = s_impact,
        driver_genes = driver_genes,
        iarc_transactivation = iarc.get('transactivation_class'),
        iarc_structure_function = iarc.get('structure_function_class'),
        iarc_hotspot = iarc.get('hotspot') == 'yes',
        iarc_experimental_gof = iarc.get('experimental_gof'),
        iarc_temperature_sensitive = iarc.get('temperature_sensitive'),
        iarc_somatic_count = iarc.get('somatic_count', 0) or 0,
        iarc_germline_count = iarc.get('germline_count', 0) or 0
    )

def evaluate_mutations(parsed_mutations):
    return [evaluate_mutation(m) for m in parsed_mutations]

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from mutationparser import parse_mutations
    mutations = parse_mutations(["R175H", "R248W"])
    for ev in evaluate_mutations(mutations):
        print(f"\n{ev.aa_change}")
        print(f"  Domain:    {ev.domain}")
        print(f"  DMS score: {ev.dms_score}")
        print(f"  Severity:  {ev.functional_severity}")
        print(f"  ClinVar:   {ev.clinvar_significance}")