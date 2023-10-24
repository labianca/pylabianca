# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:55:37 2023

@author: mmagnuski
"""
import os.path as op

import numpy as np
import pandas as pd
from scipy.io import loadmat

import pylabianca as pln


# we read matlab file with events
data_dir = r'C:\Users\Asus\Dropbox\Sarenka\PROJECTS\pylabianca\ft_data'
event_fname = 'p029_sort_final_01_events.mat'
events_mat = loadmat(op.join(data_dir, event_fname),
                     squeeze_me=True)['event']

# and then format the events into mne-python events array
event_time = events_mat['timestamp'].astype('int64')
event_id = events_mat['value'].astype('int64')
del events_mat

# %%
# target changes first: 20001, 20003
# distractor changes first: 20002, 20004
# # so attention in RF: 20001 and 20002
# and attention out of RF is 20003, 20004

events = {'begin trial': 10044, 'end trial': 10045, 'stim on': 10030,
          'target change': 12001, 'distractor change': 12000,
          'correct response': 10041}
attCnds = np.arange(20001, 20004 + 1)

begmark = np.where(event_id == events['begin trial'])[0]
endmark = np.where(event_id == events['end trial'])[0]

assert len(begmark) == len(endmark)
assert (endmark > begmark).all()

columns=['has_stimon', 'has_targetchange', 'has_distractorchange',
         'condition', 'correct', 'change_first', 'attention']
df = pd.DataFrame(columns=columns)

for idx in range(len(begmark)):
    trigs = event_id[begmark[idx]:endmark[idx] + 1]
    trig_ts = event_time[begmark[idx]:endmark[idx] + 1]
    t_start = trig_ts[0] / 40_000

    which_stimon = np.where(trigs == events['stim on'])[0]
    n_stm = len(which_stimon)
    has_stimon = n_stm > 0
    df.loc[idx, 'has_stimon'] = has_stimon

    tchange_idx = np.where(trigs == events['target change'])[0]
    dchange_idx = np.where(trigs == events['distractor change'])[0]

    df.loc[idx, 'has_targetchange'] = len(tchange_idx) > 0
    df.loc[idx, 'has_distractorchange'] = len(dchange_idx) > 0

    # condition
    cond_idx = np.where(np.isin(trigs, attCnds))[0][0]
    cond_val = trigs[cond_idx]
    change_first = 'target' if (cond_val % 2) == 1 else 'distractor'
    att_cnd = 'in' if cond_val < 20003 else 'out'
    is_correct = events['correct response'] in trigs
    df.loc[idx, 'condition'] = cond_val
    df.loc[idx, 'change_first'] = change_first
    df.loc[idx, 'attention'] = att_cnd
    df.loc[idx, 'correct'] = is_correct

# %%
df = df.infer_objects()
df.to_csv(op.join(data_dir, 'monkey_stim.csv'), index=False)