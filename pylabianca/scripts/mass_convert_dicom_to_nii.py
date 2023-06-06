# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:03:17 2023

@author: mmagnuski
"""


# %%
import os
import os.path as op

anat_dir = r'C:\Users\mmagnuski\OneDrive - Nencki PAN\RESEARCH\anat'
source_dir = op.join(anat_dir, 'sourcedata')
scan_to_postfix = {'mri': 'T1w', 'ct': 'ct'}

subjects = ['sub-U11', 'sub-U12', 'sub-U13']

for subj in subjects:
    # make sure output directory exists
    out_dir = op.join(anat_dir, subj)

    if not op.exists(out_dir):
        os.mkdir(out_dir)

    for this_scan in ['mri', 'ct']:
        this_postfix = scan_to_postfix[this_scan]
        subj_source_dir = op.join(source_dir, subj, this_scan)
        command = f'dcm2niix -o "{out_dir}" -f {subj}_{this_postfix} "{subj_source_dir}"'

        # run the command
        os.system(command)


# %% UTILS

# check n files per folder for Wrocław anat data (folder structure sometimes is messy):
data_dir = r'C:\Users\mmagnuski\OneDrive - Nencki PAN\RESEARCH\anat\sourcedata\Wrocław'

# check subfolders that have more than 100 files
def find_folders_with_many_files(data_dir, min_files=100):
    output = dict()
    for root, dirs, files in os.walk(data_dir, topdown=False):
        if len(files) >= min_files:
            output[root] = len(files)

    return output
