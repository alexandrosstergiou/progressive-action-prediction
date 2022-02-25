import csv
import os

parent_dir = 'UCF-101'
label_files = ['trainlist01.txt', 'trainlist02.txt', 'trainlist03.txt', 'testlist01.txt', 'testlist02.txt', 'testlist03.txt']

data_dict = {1:{'train':{},'test':{}},2:{'train':{},'test':{}},3:{'train':{},'test':{}}}
for l in label_files:
    with open(os.path.join(parent_dir,l)) as l_file:
        data =l_file.readlines()
        if '1' in l:
            s = 1
        elif '2' in l:
            s = 2
        else:
            s = 3
    for dat in data:
        line = dat.strip('\n').split('/')
        c = line[0]
        id = line[-1]
        id = id.split('.avi')[0]
        if 'train' in l:
            split = 'train'
        else:
            split = 'test'
        data_dict[s][split][id] = c

    # create csv file with correct format
    with open(os.path.join(parent_dir,l.split('.')[0]+'.csv'), "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['label','id','split'])

        for key,value in data_dict[s][split].items():
            writer.writerow([value,key,split])

# create single file for both train and val
for s in data_dict.keys():
    with open(os.path.join(parent_dir,'fulllist{:02d}.csv'.format(s)), "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['label','id','split'])

        for split in data_dict[s].keys():
            for key,value in data_dict[s][split].items():
                writer.writerow([value,key,split])
