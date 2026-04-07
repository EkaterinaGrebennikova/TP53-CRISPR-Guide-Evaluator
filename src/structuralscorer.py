import os, math

PDB_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', '2OCJ.pdb')
DNA_CONTACT_RESIDUES = {120, 241, 248, 273, 276, 277, 280, 281, 283}

def parse_ca_coordinates(pdb_file: str) -> dict:
    coords = {}
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            chain     = line[21]
            if atom_name != 'CA' or chain != 'A':
                continue
            res_num = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            coords[res_num] = (x, y, z)
    return coords

def distance_3d(coord1, coord2):
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 + (coord1[2]-coord2[2])**2)

def get_structural_impact(aa_position):
    coords = parse_ca_coordinates(PDB_FILE)
    if aa_position not in coords:
        return 0.0
    mut_coord = coords[aa_position]
    min_dist = float('inf')
    for contact_res in DNA_CONTACT_RESIDUES:
        if contact_res in coords:
            d = distance_3d(mut_coord, coords[contact_res])
            if d < min_dist:
                min_dist = d
    return round(1 / (1 + min_dist / 10), 3)

if __name__ == "__main__":
    coords = parse_ca_coordinates(PDB_FILE)
    print(f"Parsed {len(coords)} residues from 2OCJ")
    for mut in [175, 248, 273, 282, 50]:
        score = get_structural_impact(mut)
        print(f"  Residue {mut}: structural impact = {score}")
