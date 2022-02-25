import csv
import os
import shutil
import pandas as pd

labels = {
    "Holding something":0,
    "Turning something upside down":0,
    "Opening something":0,
    "Picking something up":0,
    "Closing something":0,
    "Showing a photo of something to the camera":0,
    "Pushing something from left to right":0,
    "Showing something behind something":0,
    "Putting something into something":0,
    "Pretending to turn something upside down":0,
    "Putting something similar to other things that are already on the table":0,
    "Pulling something from left to right":0,
    "Pretending to open something without actually opening it":0,
    "Putting something upright on the table":0,
    "Stuffing something into something":0,
    "Putting something on the edge of something so it is not supported and falls down":0,
    "Pretending to sprinkle air onto something":0,
    "Pulling two ends of something so that it gets stretched":0,
    "Poking a stack of something so the stack collapses":0,
    "Pulling two ends of something so that it separates into two pieces":0,
    "Poking a stack of something without the stack collapsing":0
}

files = ["something-something-v1-true_fine.csv",
         "something-something-v1-train_fine.csv",
         "something-something-v1-val_fine.csv"]

t_labels = []
t_ids = []
t_split = []
for file in files:
    with open(file) as csvfile:
        reader =csv.reader(csvfile, delimiter=',')
        for row in reader:
            label, id, split = row
            if label in labels.keys():
                labels[label] = labels[label] + 1
                t_labels.append(label)
                t_ids.append(id)
                t_split.append(split)


    file_dict = {'label':t_labels,
                  'id':t_ids,
                  'split':t_split}
    dst_file = file.split('_fine')[0]+'_sub21.csv'

    (pd.DataFrame.from_dict(data=file_dict).to_csv(dst_file, header=True, sep=',', index=False))
