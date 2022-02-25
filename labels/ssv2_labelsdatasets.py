import csv
import os
import shutil

base_dir = '20bn-something-something-v2'
fine_based_dir = 'ssv2'

with open('ssv2_true.csv') as csvfile:
    reader =csv.reader(csvfile, delimiter=',')
    for row in reader:
        if 'split' in row[2]:
            continue
        file = os.path.join(base_dir,row[1]+'.webm')
        dst_dir = os.path.join(fine_based_dir,row[0])
        print(dst_dir)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)

        shutil.copy2(file,os.path.join(dst_dir))
