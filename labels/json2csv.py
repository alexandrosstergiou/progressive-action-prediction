import json
import pandas as pd

files = [
"something-something-v2-train.json",
"something-something-v2-validation.json"
]

true_ids = {}


for file in files:

    set_ids = {}

    # Opening JSON file
    f = open(file)

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Iterating through the json
    # list
    for entry in data:
        id = entry['id']
        label = entry['label']
        template = entry['template'].replace('[','').replace(']','').replace(',','')
        if 'train' in file:
            set = 'train'
        elif 'validation' in file:
            set = 'val'
        else:
            set = 'test'
        set_ids[id] = [template, id, set]
        true_ids[id] = [template, id, set]

    set_ids = pd.DataFrame.from_dict(set_ids, orient='index', columns=['label', 'id', 'split']).sort_values(by=['label'])
    set_ids.to_csv(file.split('.')[0]+'.csv', index=False, header=True)

true_ids = pd.DataFrame.from_dict(true_ids, orient='index', columns=['label', 'id', 'split']).sort_values(by=['label'])
true_ids.to_csv('something-something-v2-true.csv', index=False, header=True)
