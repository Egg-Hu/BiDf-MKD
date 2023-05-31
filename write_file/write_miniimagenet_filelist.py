import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import csv
import os
data_path='../DFL2Ldata/Miniimagenet'
savedir = '../DFL2Ldata/Miniimagenet/split/'

os.makedirs(savedir, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']

if __name__=='__main__':
    for split in split_list:
        class_file_list=[]
        path_list = []
        label_list = []
        split_path=join(data_path,split)
        class_list=[f for f in listdir(split_path) if isdir(join(split_path, f))]
        for class_name in class_list:
            class_path=join(split_path,class_name)
            class_file_list.append([ join(class_path, cf) for cf in listdir(class_path) if (isfile(join(class_path,cf)) and cf[0] != '.')])
        for i, file_list in enumerate(class_file_list):
            path_list=path_list+file_list
            context = file_list[0].split('/')[-2]
            label_list = label_list + np.repeat(context, len(file_list)).tolist()


        fo = open(savedir + split + ".csv", "w",newline='')
        writer = csv.writer(fo)
        writer.writerow(['filename','label'])
        temp=np.array(list(zip(path_list,label_list)))
        writer.writerows(temp)
        fo.close()
        print("%s -OK" %split)
