import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
SPLIT_PATH = osp.join('../DFL2Ldata/Miniimagenet/split')


def identity(x):
    return x

class MiniImageNet(Dataset):
    """ Usage:
    """
    def __init__(self, setname, augment=False,noTransform=False):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        #print('csv_path:',csv_path)
        self.data, self.label = self.parse_csv(csv_path, setname)
        self.num_class = len(set(self.label))

        self.img_size = 84
        if augment and setname == 'meta_train':
            transforms_list = [
                  transforms.Resize((84, 84)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
        else:
            transforms_list = [
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
        if noTransform==True:
            self.transform=lambda x:np.asarray(x)
        else:
            self.transform = transforms.Compose(
                transforms_list
            )

    def parse_csv(self, csv_path, setname):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []#[path0,path2,path2,...]
        label = []#[0,0,0,1,2,...]
        lb = -1

        self.wnids = []

        # for l in tqdm(lines, ncols=64):
        for l in lines:
            name, wnid = l.split(',')
            path = name
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append( path )
            label.append(lb)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        image = self.transform(Image.open(data).convert('RGB'))

        return image, label

class MiniImageNet_Specific(Dataset):
    """ Usage:
    """

    def __init__(self, setname, specific=None, augment=False, mode='all',noTransform=False):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        self.data, self.label = self.parse_csv(csv_path, setname)
        self.num_class = len(set(self.label))
        if mode == 'all':
            data = [z[0] for z in zip(self.data, self.label) if z[1] in specific]
            label = [z[1] for z in zip(self.data, self.label) if z[1] in specific]
        elif mode == 'train':
            data=[]
            label=[]
            for select_class in specific:
                data_specific=[]
                label_specific=[]
                for z in zip(self.data, self.label):
                    if z[1] ==select_class:
                        data_specific.append(z[0])
                        label_specific.append(z[1])
                data_specific=data_specific[:int(len(data_specific)*0.8)]
                label_specific=label_specific[:int(len(label_specific)*0.8)]
                data.append(data_specific)
                label.append(label_specific)
            data=[j for i in data for j in i ]
            label=[j for i in label for j in i ]

            self.data=data
            self.label=label
        elif mode == 'test':
            data = []
            label = []
            for select_class in specific:
                data_specific = []
                label_specific = []
                for z in zip(self.data, self.label):
                    if z[1] == select_class:
                        data_specific.append(z[0])
                        label_specific.append(z[1])
                data_specific = data_specific[int(len(data_specific) * 0.8):]
                label_specific = label_specific[int(len(label_specific) * 0.8):]
                data.append(data_specific)
                label.append(label_specific)
            data = [j for i in data for j in i]
            label = [j for i in label for j in i]
            self.data = data
            self.label = label
        self.data = data
        self.label = label
        self.num_class = len(set(self.label))

        self.img_size = 84
        if augment and setname == 'meta_train':
            transforms_list = [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
        else:
            transforms_list = [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
        if noTransform==True:
            self.transform=lambda x:np.asarray(x)
        else:
            self.transform = transforms.Compose(
                transforms_list
            )

    def parse_csv(self, csv_path, setname):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []#[path0,path2,path2,...]
        label = []#[0,0,0,1,2,...]
        lb = -1

        self.wnids = []

        # for l in tqdm(lines, ncols=64):
        for l in lines:
            name, wnid = l.split(',')
            path = name
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append( path )
            label.append(lb)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        image = self.transform(Image.open(data).convert('RGB'))

        return image, label
if __name__ == "__main__" :
    dataset=MiniImageNet(setname='meta_train', augment=False,noTransform=True)
    data,label=dataset.__getitem__(1)
    print(data)