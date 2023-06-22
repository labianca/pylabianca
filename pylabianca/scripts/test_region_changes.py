# -*- coding: utf-8 -*-
"""
Check region changes after localization.

Created on Tue Jun 20 20:01:40 2023

@author: Asus
"""

import os
import os.path as op

import numpy as np
import pandas as pd


data_dir = r'C:\Users\Asus\OneDrive - Nencki PAN\RESEARCH\anat\derivatives\labels'
files = [f for f in os.listdir(data_dir) if f.endswith('.tsv')]
print(files)


# %%


# %% compare along closest_anat
# SMA -> ACC, ACC -> SMA, SMA -- SMA, ACC -- ACC, + SMA, + ACC

region_transl = {'ACC': 'Anterior-Cingulate', 'SMA': 'Superior-Frontal'}

orig_reg = list(region_transl.keys())
new_reg = list(region_transl.values())
inverse_translate = {val: key for key, val in region_transl.items()}

def classify_row(row):
    case = ''
    this_label = row.label[1:]
    in_orig = this_label in orig_reg
    comp_new = [nw in row.closest_anat for nw in new_reg]
    in_new = any(comp_new)
    
    if not in_new and not in_orig:
        return None

    if in_new:
        matching = new_reg[np.where(comp_new)[0][0]]
        new_old_label = inverse_translate[matching]
        
        if in_orig:
            rel = '--' if this_label == new_old_label else '->'
            case = f'{this_label} {rel} {new_old_label}'
        else:
            case = f'+ {new_old_label}'
    else:
        if in_orig:
            case = f'- {this_label}'
    
    return case


def add_case_to_dict(case, dct):
    if case in dct:
        dct[case] += 1
    else:
        dct[case] = 1


# %%
per_subject = dict()
total = dict()

for fname in files:
    this_subject = dict()
    sub = fname.split('_')[0]

    df = pd.read_csv(op.join(data_dir, fname), sep='\t')
    n_cols = df.shape[0]
    
    for idx in range(n_cols):
        case = classify_row(df.iloc[idx, :])
        
        if case is not None:
            add_case_to_dict(case, this_subject)
            add_case_to_dict(case, total)
    
    per_subject[sub] = this_subject
