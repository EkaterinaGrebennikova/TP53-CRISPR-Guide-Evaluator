import json, os

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'dms_scores.json')
_cache = None

def load_dms_scores() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE) as f:
        _cache = json.load(f)
    return _cache

def get_dms_score(aa_change):
    scores = load_dms_scores()
    return scores.get(aa_change, None)

if __name__ == "__main__":
    print("DATA_FILE:", DATA_FILE, "exists:", os.path.exists(DATA_FILE))
    for mut in ["R175H", "R248W", "R273H", "G245S", "R282W", "FAKE123"]:
        score = get_dms_score(mut)
        print(f"{mut}: {score}")