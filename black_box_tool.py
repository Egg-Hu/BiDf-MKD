import os.path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.cifar100 import Cifar100, Cifar100_Specific
from dataset.samplers import CategoriesSampler
from dataset.miniimagenet import MiniImageNet, MiniImageNet_Specific
from dataset.omniglot import Omniglot, Omniglot_Specific
import network
from dataset.cub import CUB, CUB_Specific
from network import Conv4, ResNet34, ResNet18, ResNet50, ResNet10, ResNet101
import torch.nn.functional as F
from torchvision import transforms


def get_dataloader(args,noTransform_test=False):
    if args.dataset == 'cifar100':
        trainset = Cifar100(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size=trainset.img_size
        # train_sampler = CategoriesSampler(trainset.label,
        #                                   args.episode_train,
        #                                   args.way_train,
        #                                   args.num_sup_train + args.num_qur_train,maml_allclass=args.maml_allclass)
        # train_loader = DataLoader(dataset=trainset,
        #                           num_workers=8,
        #                           batch_sampler=train_sampler,
        #                           pin_memory=True)
        valset=Cifar100(setname='meta_val', augment=False)
        val_sampler = CategoriesSampler(valset.label,
                                          args.episode_test,
                                          args.way_test,
                                          args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                  num_workers=8,
                                  batch_sampler=val_sampler,
                                  pin_memory=True)
        testset = Cifar100(setname='meta_test', augment=False,noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                num_workers=8,
                                batch_sampler=test_sampler,
                                pin_memory=True)
        return None, val_loader, test_loader
    elif args.dataset == 'miniimagenet':
        trainset = MiniImageNet(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        # train_sampler = CategoriesSampler(trainset.label,
        #                                   args.episode_train,
        #                                   args.way_train,
        #                                   args.num_sup_train + args.num_qur_train)
        # train_loader = DataLoader(dataset=trainset,
        #                           num_workers=8,
        #                           batch_sampler=train_sampler,
        #                           pin_memory=True)
        valset = MiniImageNet(setname='meta_val', augment=False)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = MiniImageNet(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return None, val_loader, test_loader
    elif args.dataset == 'omniglot':
        trainset = Omniglot(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        # train_sampler = CategoriesSampler(trainset.label,
        #                                   args.episode_train,
        #                                   args.way_train,
        #                                   args.num_sup_train + args.num_qur_train)
        # train_loader = DataLoader(dataset=trainset,
        #                           num_workers=8,
        #                           batch_sampler=train_sampler,
        #                           pin_memory=True)
        testset = Omniglot(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        val_loader=None
        return None, val_loader, test_loader
    elif args.dataset=='cub':
        trainset = CUB(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        # train_sampler = CategoriesSampler(trainset.label,
        #                                   args.episode_train,
        #                                   args.way_train,
        #                                   args.num_sup_train + args.num_qur_train)
        # train_loader = DataLoader(dataset=trainset,
        #                           num_workers=8,
        #                           batch_sampler=train_sampler,
        #                           pin_memory=True)
        valset = CUB(setname='meta_val', augment=False)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = CUB(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return None, val_loader, test_loader
    elif args.dataset=='mix':
        testset_cifar = Cifar100(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset_cifar.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader_cifar = DataLoader(dataset=testset_cifar,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        testset_mini = MiniImageNet(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset_mini.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader_mini = DataLoader(dataset=testset_mini,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)

        testset_cub = CUB(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset_cub.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader_cub = DataLoader(dataset=testset_cub,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return test_loader_cifar, test_loader_mini, test_loader_cub
    else:
        ValueError('not implemented!')
    #return None, val_loader, test_loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False

def set_maml(flag):
    network.ConvBlock.maml = flag
    network.SimpleBlock.maml = flag
    network.BottleneckBlock.maml = flag
    network.ResNet.maml = flag
    network.ConvNet.maml = flag

def get_model(args,mode='train',set_maml_value=True,arbitrary_input=False):
    set_maml(set_maml_value)
    if mode=='train':
        way=args.way_train
    else:
        way = args.way_test
    if args.backbone == 'conv4':
        model_maml = Conv4(flatten=True, out_dim=way, img_size=args.img_size,arbitrary_input=arbitrary_input,channel=args.channel)
    elif args.backbone == 'resnet34':
        model_maml = ResNet34(flatten=True, out_dim=way)
    elif args.backbone == 'resnet18':
        model_maml = ResNet18(flatten=True, out_dim=way)
    elif args.backbone == 'resnet50':
        model_maml = ResNet50(flatten=True, out_dim=way)
    elif args.backbone == 'resnet101':
        model_maml = ResNet101(flatten=True, out_dim=way)
    elif args.backbone=='resnet10':
        model_maml = ResNet10(flatten=True, out_dim=way)
    else:
        raise NotImplementedError
    return model_maml
def get_premodel(args,getType=None):
    set_maml(False)
    way=args.way_pretrain
    if args.pre_backbone == 'conv4':
        model_maml = Conv4(flatten=True, out_dim=way, img_size=args.img_size,arbitrary_input=False,channel=args.channel)
    elif args.pre_backbone == 'resnet34':
        model_maml = ResNet34(flatten=True, out_dim=way)
    elif args.pre_backbone == 'resnet18':
        model_maml = ResNet18(flatten=True, out_dim=way)
    elif args.pre_backbone == 'resnet50':
        model_maml = ResNet50(flatten=True, out_dim=way)
    elif args.backbone == 'resnet101':
        model_maml = ResNet101(flatten=True, out_dim=way)
    elif args.pre_backbone=='resnet10':
        model_maml = ResNet10(flatten=True, out_dim=way)
    elif args.pre_backbone=='mix':
        pre_backbone = getType
        if pre_backbone == 'conv4':
            model_maml = Conv4(flatten=True, out_dim=way, img_size=args.img_size)
        elif pre_backbone == 'resnet34':
            model_maml = ResNet34(flatten=True, out_dim=way)
        elif pre_backbone == 'resnet18':
            model_maml = ResNet18(flatten=True, out_dim=way)
        elif pre_backbone == 'resnet50':
            model_maml = ResNet50(flatten=True, out_dim=way)
        elif pre_backbone == 'resnet10':
            model_maml = ResNet10(flatten=True, out_dim=way)
    else:
        ValueError('not implemented!')
    return model_maml
class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)
def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm
def data2supportquery(args,mode,data):
    if mode=='train':
        way=args.way_train
        num_sup=args.num_sup_train
        num_qur=args.num_qur_train
    else:
        way = args.way_test
        num_sup = args.num_sup_test
        num_qur = args.num_qur_test
    label = torch.arange(way, dtype=torch.int16).repeat(num_qur+num_sup)
    label = label.type(torch.LongTensor)
    label = label.cuda()
    support=data[:way*num_sup]
    support_label=label[:way*num_sup]
    query=data[way*num_sup:]
    query_label=label[way*num_sup:]
    return support,support_label,query,query_label
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
NORMALIZE_DICT = {
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar10': dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'cifar100': dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    'miniimagenet': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'cub': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'tinyimagenet': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    'cub200': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'stanford_dogs': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'stanford_cars': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365_32x32': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365_64x64': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'places365': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'svhn': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'tiny_imagenet': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'imagenet_32x32': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # for semantic segmentation
    'camvid': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'nyuv2': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)
def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)
def label_abs2relative(specific, label_abs):
    trans = dict()
    for relative, abs in enumerate(specific):
        trans[abs] = relative
    label_relative = []
    for abs in label_abs:
        label_relative.append(trans[abs.item()])
    return torch.LongTensor(label_relative)
def pretrain(args,specific,device):
    if args.dataset=='cifar100':
        train_dataset = Cifar100_Specific(setname='meta_train', specific=specific, mode='train')
        assert len(train_dataset)==args.way_pretrain*480, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
        test_dataset = Cifar100_Specific(setname='meta_train', specific=specific, mode='test')
        assert len(test_dataset) == args.way_pretrain*120, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
        channel=3
        num_epoch = 60
        learning_rate = 0.01
    elif args.dataset=='miniimagenet':
        train_dataset = MiniImageNet_Specific(setname='meta_train', specific=specific, mode='train')
        assert len(train_dataset) == args.way_pretrain*480, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        test_dataset = MiniImageNet_Specific(setname='meta_train', specific=specific, mode='test')
        assert len(test_dataset) == args.way_pretrain*120, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                  pin_memory=True)
        channel = 3
        num_epoch = 60
        learning_rate = 0.01
    elif args.dataset=='omniglot':
        train_dataset = Omniglot_Specific(setname='meta_train', specific=specific, mode='train')
        assert len(train_dataset) == 80, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=80, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        test_dataset = Omniglot_Specific(setname='meta_train', specific=specific, mode='test')
        assert len(test_dataset) == 20, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=20, shuffle=True, num_workers=8,
                                                  pin_memory=True)
        channel = 1
        num_epoch = 60
        learning_rate = 0.1
    elif args.dataset=='cub':
        train_dataset = CUB_Specific(setname='meta_train', specific=specific, mode='train')
        #assert len(train_dataset) == 2400, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        test_dataset = CUB_Specific(setname='meta_train', specific=specific, mode='test')
        #assert len(test_dataset) == 600, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                  pin_memory=True)
        channel = 3
        num_epoch = 100
        learning_rate = 0.01
    set_maml(False)
    if args.pre_backbone=='conv4':
        teacher=Conv4(flatten=True, out_dim=args.way_pretrain, img_size=train_dataset.img_size,arbitrary_input=False,channel=channel).cuda(device)
        optimizer = torch.optim.Adam(params=teacher.parameters(), lr=learning_rate)
        #optimizer=torch.optim.SGD(params=teacher.parameters(),lr=learning_rate,momentum=.9, weight_decay=5e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[70], gamma=0.2)#70 default#[30, 50, 80]
    elif args.pre_backbone=='resnet18':
        teacher=ResNet18(flatten=True,out_dim=args.way_pretrain).cuda(device)
        #optimizer = torch.optim.Adam(params=teacher.parameters(), lr=learning_rate)
        optimizer=torch.optim.SGD(params=teacher.parameters(),lr=learning_rate,momentum=.9, weight_decay=5e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.2)
    elif args.pre_backbone=='resnet10':
        teacher = ResNet10(flatten=True, out_dim=args.way_pretrain).cuda(device)
        # optimizer = torch.optim.Adam(params=teacher.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(params=teacher.parameters(), lr=learning_rate, momentum=.9, weight_decay=5e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.2)
    elif args.pre_backbone=='resnet50':
        teacher = ResNet50(flatten=True, out_dim=args.way_pretrain).cuda(device)
        # optimizer = torch.optim.Adam(params=teacher.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(params=teacher.parameters(), lr=learning_rate, momentum=.9, weight_decay=5e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.2)
    elif args.pre_backbone=='resnet101':
        teacher = ResNet101(flatten=True, out_dim=args.way_pretrain).cuda(device)
        # optimizer = torch.optim.Adam(params=teacher.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(params=teacher.parameters(), lr=learning_rate, momentum=.9, weight_decay=5e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.2)
    #train
    best_pre_model=None
    best_acc=None
    not_increase=0
    for epoch in range(num_epoch):
        # train
        teacher.train()
        for batch_count, batch in enumerate(train_loader):
            optimizer.zero_grad()
            image, abs_label = batch[0].cuda(device), batch[1].cuda(device)
            relative_label = label_abs2relative(specific=specific, label_abs=abs_label).cuda(device)
            logits = teacher(image)
            criteria = torch.nn.CrossEntropyLoss()
            loss = criteria(logits, relative_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 50)
            optimizer.step()
        lr_schedule.step()
        correct, total = 0, 0
        teacher.eval()
        for batch_count, batch in enumerate(test_loader):
            image, abs_label = batch[0].cuda(device), batch[1].cuda(device)
            relative_label = label_abs2relative(specific=specific, label_abs=abs_label).cuda(device)
            logits = teacher(image)
            prediction = torch.max(logits, 1)[1]
            correct = correct + (prediction.cpu() == relative_label.cpu()).sum()
            total = total + len(relative_label)
        test_acc = 100 * correct / total
        if best_acc==None or best_acc<test_acc:
            best_acc=test_acc
            best_epoch=epoch
            best_pre_model=teacher.state_dict()
            not_increase=0
        else:
            not_increase=not_increase+1
            if not_increase==60:#7 for cifar and mini; 20 for omniglot
                print('early stop at:',best_epoch)
                break
        print('epoch{}acc:'.format(epoch),test_acc,'best{}acc:'.format(best_epoch),best_acc)

    return best_pre_model,best_acc
def pretrains(args,num,device,pretrain_path):
    list_all=[]
    for i in range(num):
        #setup_seed(222 + i)
        specific=random.sample(range(args.class_num),args.way_pretrain)
        print(specific)
        teacher,acc=pretrain(args,specific,device)
        print('teacher{}_acc:'.format(i),acc)
        list_all.append([specific,acc])
        torch.save({'teacher':teacher,'specific':specific},os.path.join(pretrain_path,'model_specific_{}.pth'.format(i)))


def get_transform(args,dataset=None):
    if dataset==None:
        dataset=args.dataset
    transform=None
    if dataset=='cifar100':
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
    elif dataset=='miniimagenet':
        transform = transforms.Compose(
            [
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif dataset=='omniglot':
        transform = transforms.Compose(
            [
                #transforms.Resize((28, 28)),
                lambda x: x.resize((28, 28)),
                lambda x: np.reshape(x, (28, 28, 1)),
                transforms.ToTensor(),
            ]
        )
    elif dataset=='cub':
        transform = transforms.Compose(
            [
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    return transform


def get_transform_no_toTensor(args,dataset=None):
    if dataset==None:
        dataset=args.dataset
    transform = None
    if dataset=='cifar100':
        transform = transforms.Compose(
            [
                #transforms.Resize((32, 32)),
                transforms.RandomCrop(size=[32, 32], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
    elif dataset=='miniimagenet':
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=[84, 84], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif dataset=='omniglot':
        transform = transforms.Compose(
            [
                #lambda x: x.resize((28, 28), padding=4),
                transforms.RandomCrop((28, 28), padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif dataset=='cub':
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=[84, 84], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    return transform
# def get_transform_no_toTensor_localview(args):
#     if args.dataset=='cifar100':
#         transform = transforms.Compose(
#             [
#                 #transforms.Resize((32, 32)),
#                 transforms.RandomResizedCrop(size=[32, 32], scale=(0.25, 1.0)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
#             ]
#         )
#     elif args.dataset=='miniimagenet':
#         transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(size=[84, 84], scale=(0.25, 1.0)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             ]
#         )
#     elif args.dataset=='omniglot':
#         pass
#     return transform
#
#
#
# def get_transform_globalview(args):
#     if args.dataset=='cifar100':
#         transform = transforms.Compose(
#             [
#                 transforms.RandomCrop(size=[32, 32], padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
#             ]
#         )
#     elif args.dataset=='miniimagenet':
#         transform = transforms.Compose(
#             [
#                 transforms.RandomCrop(size=[84, 84],padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             ]
#         )
#     elif args.dataset=='omniglot':
#         pass
#     return transform
# def get_transform_localview(args):
#     if args.dataset=='cifar100':
#         transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(size=[32, 32], scale=(0.25, 1.0)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
#             ]
#         )
#     elif args.dataset=='miniimagenet':
#         transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(size=[84, 84], scale=(0.25, 1.0)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             ]
#         )
#     elif args.dataset=='omniglot':
#         pass
#     return transform


def one_hot(label_list,class_num):
    temp_label=label_list.reshape(len(label_list),1)
    y_one_hot = torch.zeros(len(label_list), class_num).scatter_(1, temp_label, 1)
    return y_one_hot


def get_84_transform(args,dataset=None):
    if dataset==None:
        dataset=args.dataset
    transform=None
    if dataset=='cifar100':
        transform = transforms.Compose(
            [
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
    elif dataset=='miniimagenet':
        transform = transforms.Compose(
            [
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif dataset=='omniglot':
        transform = transforms.Compose(
            [
                #transforms.Resize((28, 28)),
                lambda x: x.resize((28, 28)),
                lambda x: np.reshape(x, (28, 28, 1)),
                transforms.ToTensor(),
            ]
        )
    elif dataset=='cub':
        transform = transforms.Compose(
            [
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    return transform