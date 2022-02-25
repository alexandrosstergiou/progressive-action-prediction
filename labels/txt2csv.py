import csv
import os
import pandas as pd

def load_train_test(train_file, val_file):
    with open(train_file) as f:
        t_lines = f.readlines()
        t_lines = [line.rstrip() for line in t_lines]

    with open(val_file) as f:
        v_lines = f.readlines()
        v_lines = [line.rstrip() for line in v_lines]

    t_vids = []
    t_labels = []
    for line in t_lines:
        label, vid = line.split('/')
        vid = vid.split('.avi')[0]
        t_vids.append(vid)
        t_labels.append(label)

    v_vids = []
    v_labels = []
    for line in v_lines:
        label, vid = line.split('/')
        vid = vid.split('.avi')[0]
        v_vids.append(vid)
        v_labels.append(label)

    train_file = train_file.split('.')[0]+'.csv'
    val_file = val_file.split('.')[0]+'.csv'
    true_file = train_file.split('train')[0]+'truelist.csv'

    train_dict = {'label':t_labels,
                  'id':t_vids,
                  'split':['train' for _ in t_labels]}

    val_dict = {'label':v_labels,
                'id':v_vids,
                'split':['val' for _ in v_labels]}

    true_dict = {'label':t_labels + v_labels,
                'id':t_vids + v_vids,
                'split':['train' for _ in t_labels] + ['val' for _ in v_labels]}


    (pd.DataFrame.from_dict(data=train_dict).to_csv(train_file, header=True, sep=',', index=False))
    (pd.DataFrame.from_dict(data=val_dict).to_csv(val_file, header=True, sep=',', index=False))
    (pd.DataFrame.from_dict(data=true_dict).to_csv(true_file, header=True, sep=',', index=False))


file_pairs = [['UCF-101/trainlist01.txt','UCF-101/testlist01.txt'],
              ['UCF-101/trainlist02.txt','UCF-101/testlist02.txt'],
              ['UCF-101/trainlist03.txt','UCF-101/testlist03.txt']]
for f_pair in file_pairs:
    load_train_test(f_pair[0],f_pair[1])
