import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import random
import csv

data_path='../DFL2Ldata/CUB_200_2011/images'
savedir = '../DFL2Ldata/CUB_200_2011/split/'
os.makedirs(savedir, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])


for split in split_list:
    num=0
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
            
        if 'meta_train' in split:
            if i % 2 ==0:
                file_list = file_list + classfile_list
                context = classfile_list[0].split('/')[-2]
                label_list = label_list + np.repeat(context, len(classfile_list)).tolist()
                num=num+1
        if 'meta_val' in split:
            if i % 4 == 1:
                file_list = file_list + classfile_list
                context = classfile_list[0].split('/')[-2]
                label_list = label_list + np.repeat(context, len(classfile_list)).tolist()
                num = num + 1
        if 'meta_test' in split:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                context = classfile_list[0].split('/')[-2]
                label_list = label_list + np.repeat(context, len(classfile_list)).tolist()
                num = num + 1
    print('split_num:',num)

    fo = open(savedir + split + ".csv", "w",newline='')
    writer = csv.writer(fo)
    writer.writerow(['filename','label'])
    temp=np.array(list(zip(file_list,label_list)))
    writer.writerows(temp)
    fo.close()
    print("%s -OK" %split)
