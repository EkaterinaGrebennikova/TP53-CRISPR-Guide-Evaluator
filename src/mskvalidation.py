import os
import requests
import pandas as pd
from tcgaallelic import get_allelic_context
from survivalanalysis import build_survival_df

API = "https://www.cbioportal.org/api"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

TP53_ENTREZ = 7157
MDM2_ENTREZ = 4193

STUDY_ID = "msk_impact_2017"

MUTATIONS_OUT = os.path.join(DATA_DIR, 'msk_mutations.txt')
CNA_OUT       = os.path.join(DATA_DIR, 'msk_cna.txt')
CLINICAL_OUT  = os.path.join(DATA_DIR, 'msk_clinical.txt')


def _sample_to_patient(sample_id):
    # MSK-IMPACT sample barcode: P-0000001-T01-IM3 -> patient: P-0000001
    parts = sample_id.split('-')
    return '-'.join(parts[:2]) if len(parts) >= 2 else sample_id


def fetch_msk_mutations(entrez_ids):
    profile_id = f"{STUDY_ID}_mutations"
    sample_list_id = f"{STUDY_ID}_all"
    url = f"{API}/molecular-profiles/{profile_id}/mutations/fetch"
    body = {"entrezGeneIds": entrez_ids, "sampleListId": sample_list_id}
    r = requests.post(url, json=body, params={"projection": "DETAILED"}, timeout=120)
    r.raise_for_status()
    muts = r.json()

    records = []
    for m in muts:
        gene = m.get('gene') or {}
        sample_id = m.get('sampleId')
        protein_change = m.get('proteinChange') or ''
        aa_change = protein_change.replace('p.', '', 1)
        records.append({
            'patient_id':             _sample_to_patient(sample_id),
            'sample_id':              sample_id,
            'Hugo_Symbol':            gene.get('hugoGeneSymbol'),
            'aa_change':              aa_change,
            'HGVSp_Short':            protein_change,
            'Variant_Classification': m.get('mutationType'),
            'Chromosome':             m.get('chr'),
            'Start_Position':         m.get('startPosition'),
            'Reference_Allele':       m.get('referenceAllele'),
            'Tumor_Seq_Allele2':      m.get('variantAllele'),
            't_ref_count':            m.get('tumorRefCount'),
            't_alt_count':            m.get('tumorAltCount'),
        })
    print(f"  {STUDY_ID}: {len(records)} mutations")
    return pd.DataFrame(records)


def fetch_msk_cna(entrez_ids):
    profile_id = f"{STUDY_ID}_cna"
    sample_list_id = f"{STUDY_ID}_all"
    url = f"{API}/molecular-profiles/{profile_id}/molecular-data/fetch"
    body = {"entrezGeneIds": entrez_ids, "sampleListId": sample_list_id}
    r = requests.post(url, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()

    by_sample = {}
    for d in data:
        sample = d['sampleId']
        gene   = d['entrezGeneId']
        val    = d['value']
        if sample not in by_sample:
            by_sample[sample] = {
                'patient_id': _sample_to_patient(sample),
                'sample_id':  sample,
                'tp53_cna':   0,
                'mdm2_cna':   0,
            }
        try:
            v = int(val)
        except (TypeError, ValueError):
            v = 0
        if gene == TP53_ENTREZ:
            by_sample[sample]['tp53_cna'] = v
        elif gene == MDM2_ENTREZ:
            by_sample[sample]['mdm2_cna'] = v
    print(f"  {STUDY_ID}: {len(by_sample)} CNA samples")
    return pd.DataFrame(by_sample.values())


def fetch_msk_clinical():
    url = f"{API}/studies/{STUDY_ID}/clinical-data"
    r = requests.get(url, params={"clinicalDataType": "PATIENT", "pageSize": 100000}, timeout=120)
    r.raise_for_status()
    patient_rows = r.json()

    r2 = requests.get(url, params={"clinicalDataType": "SAMPLE", "pageSize": 100000}, timeout=120)
    r2.raise_for_status()
    sample_rows = r2.json()

    patient_attrs = {}
    for row in patient_rows:
        pid = row.get('patientId')
        attr = row.get('clinicalAttributeId')
        val  = row.get('value')
        patient_attrs.setdefault(pid, {})[attr] = val

    sample_cancer_type = {}
    for row in sample_rows:
        pid = row.get('patientId')
        attr = row.get('clinicalAttributeId')
        val  = row.get('value')
        if attr == 'CANCER_TYPE':
            sample_cancer_type.setdefault(pid, val)

    records = []
    for pid, attrs in patient_attrs.items():
        records.append({
            'patient_id':   pid,
            'cancer_type':  sample_cancer_type.get(pid) or attrs.get('CANCER_TYPE'),
            'os_status':    attrs.get('OS_STATUS'),
            'os_months':    attrs.get('OS_MONTHS'),
            'age':          None,  # MSK-IMPACT 2017 does not release age
            'sex':          attrs.get('SEX'),
        })
    df = pd.DataFrame(records)
    df['os_months'] = pd.to_numeric(df['os_months'], errors='coerce')
    df['age']       = pd.to_numeric(df['age'], errors='coerce')
    print(f"  {STUDY_ID}: {len(df)} patients")
    return df


def fetch_and_save():
    print("Fetching MSK-IMPACT TP53 mutations...")
    m = fetch_msk_mutations([TP53_ENTREZ])
    m.to_csv(MUTATIONS_OUT, sep='\t', index=False)
    print(f"  Saved {len(m)} mutations to {MUTATIONS_OUT}\n")

    print("Fetching MSK-IMPACT TP53/MDM2 CNA...")
    c = fetch_msk_cna([TP53_ENTREZ, MDM2_ENTREZ])
    c.to_csv(CNA_OUT, sep='\t', index=False)
    print(f"  Saved {len(c)} CNA records to {CNA_OUT}\n")

    print("Fetching MSK-IMPACT clinical data...")
    cl = fetch_msk_clinical()
    cl.to_csv(CLINICAL_OUT, sep='\t', index=False)
    print(f"  Saved {len(cl)} patients to {CLINICAL_OUT}\n")
    return m, c, cl


def load_msk_mutations():
    return pd.read_csv(MUTATIONS_OUT, sep='\t')


def load_msk_cna():
    return pd.read_csv(CNA_OUT, sep='\t')


def load_msk_clinical():
    df = pd.read_csv(CLINICAL_OUT, sep='\t')
    df['os_months'] = pd.to_numeric(df['os_months'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    return df


def run_validation():
    if not os.path.exists(MUTATIONS_OUT):
        fetch_and_save()

    m = load_msk_mutations()
    c = load_msk_cna()
    cl = load_msk_clinical()

    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    msk_sdf = build_survival_df(m, c, cl)
    mut = msk_sdf[msk_sdf['tp53_mut']]
    wt  = msk_sdf[~msk_sdf['tp53_mut']]

    print(f"\n=== MSK-IMPACT Validation ===")

    # 1. TP53-mut vs WT survival
    kmf_mut = KaplanMeierFitter().fit(mut['os_months'], mut['os_event'], label='TP53-MUT')
    kmf_wt  = KaplanMeierFitter().fit(wt['os_months'], wt['os_event'], label='TP53-WT')
    lr = logrank_test(mut['os_months'], wt['os_months'],
                      event_observed_A=mut['os_event'], event_observed_B=wt['os_event'])

    med_mut = f"{kmf_mut.median_survival_time_:.1f}" if kmf_mut.median_survival_time_ != float('inf') else 'NR'
    med_wt  = f"{kmf_wt.median_survival_time_:.1f}" if kmf_wt.median_survival_time_ != float('inf') else 'NR'
    print(f"\n--- TP53-MUT vs WT (n={len(msk_sdf)} with OS data) ---")
    print(f"  TP53-MUT: n={len(mut)}, median OS = {med_mut} mo")
    print(f"  TP53-WT:  n={len(wt)}, median OS = {med_wt} mo")
    print(f"  Logrank p = {lr.p_value:.2e}")

    # 2. Biallelic vs mono-allelic vs WT
    biallelic = msk_sdf[msk_sdf['allelic_state'] == 'biallelic_mutation']
    monoallelic = msk_sdf[msk_sdf['allelic_state'].isin(
        ['heterozygous_cn_neutral', 'heterozygous_with_gain', 'loh_with_mutation'])]

    print(f"\n--- Biallelic vs Mono-allelic vs WT ---")
    groups = [('wildtype', wt), ('mono-allelic', monoallelic), ('biallelic', biallelic)]
    for label, grp in groups:
        if len(grp) < 5:
            print(f"  {label:<16} n={len(grp):>5}  (too few for KM)")
            continue
        kmf = KaplanMeierFitter().fit(grp['os_months'], grp['os_event'], label=label)
        med = f"{kmf.median_survival_time_:.1f}" if kmf.median_survival_time_ != float('inf') else 'NR'
        if label == 'wildtype':
            print(f"  {label:<16} n={len(grp):>5}  median OS = {med:>6} mo  (reference)")
        else:
            lr_g = logrank_test(grp['os_months'], wt['os_months'],
                                event_observed_A=grp['os_event'], event_observed_B=wt['os_event'])
            print(f"  {label:<16} n={len(grp):>5}  median OS = {med:>6} mo  p={lr_g.p_value:.2e}")

    # 3. Allelic distribution (with CNA limitation note)
    msk_allelic = get_allelic_context(m, c)
    print(f"\n--- Allelic Context Distribution (n={msk_allelic['total_tp53_mutant_patients']}) ---")
    print(f"  [Note: MSK-IMPACT uses discrete CNA; het+gain/LOH counts unreliable]")
    for state, count in msk_allelic['states'].items():
        frac = count / msk_allelic['total_tp53_mutant_patients'] * 100
        print(f"  {state:<26} {count:>5} ({frac:.1f}%)")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    fetch_and_save()
