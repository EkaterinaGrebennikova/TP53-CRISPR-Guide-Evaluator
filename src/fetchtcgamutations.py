# src/fetchtcgamutations.py
# One-time data retrieval utility: pulls TP53 mutations and TP53/MDM2 CNA data
# for all 32 TCGA PanCancer Atlas studies via the cBioPortal REST API.
# Usage: python src/fetchtcgamutations.py

import os
import time
import requests
import pandas as pd

API = "https://www.cbioportal.org/api"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

TP53_ENTREZ = 7157
MDM2_ENTREZ = 4193

MUTATIONS_OUT = os.path.join(DATA_DIR, 'tcga_mutations.txt')
CNA_OUT       = os.path.join(DATA_DIR, 'tcga_cna.txt')


def get_pancancer_studies():
    r = requests.get(f"{API}/studies", timeout=30)
    r.raise_for_status()
    studies = r.json()
    return sorted([
        s['studyId'] for s in studies
        if s['studyId'].endswith('_tcga_pan_can_atlas_2018')
    ])


def fetch_mutations(study_ids, entrez_ids):
    records = []
    for sid in study_ids:
        profile_id = f"{sid}_mutations"
        sample_list_id = f"{sid}_all"
        url = f"{API}/molecular-profiles/{profile_id}/mutations/fetch"
        body = {"entrezGeneIds": entrez_ids, "sampleListId": sample_list_id}
        try:
            r = requests.post(url, json=body, params={"projection": "DETAILED"}, timeout=60)
        except requests.RequestException as e:
            print(f"  {sid}: error {e}")
            continue
        if r.status_code != 200:
            print(f"  {sid}: HTTP {r.status_code} (no mutation profile?)")
            continue
        muts = r.json()
        for m in muts:
            gene = m.get('gene') or {}
            records.append({
                'Hugo_Symbol':            gene.get('hugoGeneSymbol'),
                'Tumor_Sample_Barcode':   m.get('sampleId'),
                'HGVSp_Short':            m.get('proteinChange'),
                'Variant_Classification': m.get('mutationType'),
                'Chromosome':             m.get('chr'),
                'Start_Position':         m.get('startPosition'),
                'Reference_Allele':       m.get('referenceAllele'),
                'Tumor_Seq_Allele2':      m.get('variantAllele'),
                't_ref_count':            m.get('tumorRefCount'),
                't_alt_count':            m.get('tumorAltCount'),
                'study_id':               sid,
            })
        print(f"  {sid}: {len(muts)} mutations")
        time.sleep(0.1)
    return pd.DataFrame(records)


def fetch_cna(study_ids, entrez_ids):
    rows = []
    for sid in study_ids:
        profile_id = f"{sid}_gistic"
        sample_list_id = f"{sid}_all"
        url = f"{API}/molecular-profiles/{profile_id}/molecular-data/fetch"
        body = {"entrezGeneIds": entrez_ids, "sampleListId": sample_list_id}
        try:
            r = requests.post(url, json=body, timeout=60)
        except requests.RequestException as e:
            print(f"  {sid}: error {e}")
            continue
        if r.status_code != 200:
            print(f"  {sid}: HTTP {r.status_code} (no GISTIC profile?)")
            continue
        data = r.json()
        by_sample = {}
        for d in data:
            sample = d['sampleId']
            gene   = d['entrezGeneId']
            val    = d['value']
            if sample not in by_sample:
                by_sample[sample] = {'patient_id': sample[:12], 'mdm2_cna': 0, 'tp53_cna': 0}
            try:
                v = int(val)
            except (TypeError, ValueError):
                v = 0
            if gene == TP53_ENTREZ:
                by_sample[sample]['tp53_cna'] = v
            elif gene == MDM2_ENTREZ:
                by_sample[sample]['mdm2_cna'] = v
        rows.extend(by_sample.values())
        print(f"  {sid}: {len(by_sample)} samples")
        time.sleep(0.1)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Fetching study list...")
    studies = get_pancancer_studies()
    print(f"Found {len(studies)} PanCancer Atlas studies\n")

    print("Fetching TP53 mutations...")
    mut_df = fetch_mutations(studies, [TP53_ENTREZ])
    print(f"\nTotal TP53 mutations: {len(mut_df)}")
    mut_df.to_csv(MUTATIONS_OUT, sep='\t', index=False)
    print(f"Saved to {MUTATIONS_OUT}\n")

    print("Fetching TP53 and MDM2 CNA...")
    cna_df = fetch_cna(studies, [TP53_ENTREZ, MDM2_ENTREZ])
    print(f"\nTotal CNA records: {len(cna_df)}")
    cna_df.to_csv(CNA_OUT, sep='\t', index=False)
    print(f"Saved to {CNA_OUT}")
