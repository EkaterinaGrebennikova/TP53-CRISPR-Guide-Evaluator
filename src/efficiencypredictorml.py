import os, pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
CBE_CSV = os.path.join(DATA_DIR, 'mmc2.csv')
ABE_CSV = os.path.join(DATA_DIR, 'mmc3.csv')
CBE_MODEL = os.path.join(DATA_DIR, 'be_model_cbe.pkl')
ABE_MODEL = os.path.join(DATA_DIR, 'be_model_abe.pkl')

BASES = ['A', 'C', 'G', 'T']
_cbe_model = None
_abe_model = None

def one_hot(seq):
    l = []
    for s in seq:
        if s == 'A':
            onehot = [1, 0, 0, 0]
        elif s == 'C':
            onehot = [0, 1, 0, 0]
        elif s == 'G':
            onehot = [0, 0, 1, 0]
        elif s == 'T':
            onehot = [0, 0, 0, 1]
        else:
            onehot = [0, 0, 0, 0]
        l.extend(onehot)
    return l

def extract_features(spacer, edit_position):
    spacer = spacer.upper()
    final = []
    final.extend(one_hot(spacer))
    gc_content = sum(1 for s in spacer if s in ('G', 'C')) / len(spacer)
    final.append(gc_content)
    pos_norm = (edit_position - 1) / 19.0
    final.append(pos_norm)
    i = edit_position - 1   # convert to 0-based
    left  = spacer[i - 1] if i > 0 else 'N'
    mid   = spacer[i]
    right = spacer[i + 1] if i < len(spacer) - 1 else 'N'
    final.extend(one_hot(left))
    final.extend(one_hot(mid))
    final.extend(one_hot(right))
    return final