import csv, os
COSMIC_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'cosmic_cancer_genes.csv')

def load_cancer_genes() -> list:
    genes = []
    with open(COSMIC_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            location = row['Genome Location']
            if not location:
                continue
            try:
                chrom, coords = location.split(':')
                start, end = coords.split('-')
                genes.append({
                    'gene': row['Gene Symbol'],
                    'chrom': chrom,
                    'start': int(start),
                    'end': int(end)
                })
            except:
                continue
    return genes

def check_driver_overlap(genomic_coords):
    l = []
    genes = load_cancer_genes()
    for position in genomic_coords:
        chrom = str(position.get('chrom', ''))
        pos = position.get('pos')
        for gene in genes:
            if gene['chrom'] == chrom and gene['start'] <= pos <= gene['end'] and gene['gene'] not in l:
                l.append(gene['gene'])
    return l

if __name__ == "__main__":
    genes = load_cancer_genes()
    print(f"Loaded {len(genes)} cancer driver genes")
    # test with a known TP53 coordinate on chr17
    test = [{"chrom": "17", "pos": 7675088}]
    overlaps = check_driver_overlap(test)
    print(f"Overlaps at chr17:7675088: {overlaps}")