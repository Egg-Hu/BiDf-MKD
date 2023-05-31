import argparse
import os
from datetime import datetime
import random
import shutil
import torch.nn.functional as F
import torch
from torch import nn

from black_box_tool import get_model, get_premodel, get_transform, \
    get_transform_no_toTensor, \
    label_abs2relative, get_dataloader, data2supportquery, Timer, setup_seed, compute_confidence_interval, Generator, \
    pretrains
from methods.maml import Maml, MamlKD
from synthesis.contrastive import BlackBoxSynthesizer


parser = argparse.ArgumentParser(description='blackboxDFML')
#basic
parser.add_argument('--multigpu', type=str, default='0', help='seen gpu')
parser.add_argument('--gpu', type=int, default=0, help="gpu")
parser.add_argument('--dataset', type=str, default='cifar100', help='cifar100/miniimagenet/cub')
parser.add_argument('--pretrained_path_prefix', type=str, default='./pretrained_blackbox', help='user-defined')
#memory
parser.add_argument('--way_train', type=int, default=5, help='way')
parser.add_argument('--num_sup_train', type=int, default=5)
parser.add_argument('--num_qur_train', type=int, default=15)
parser.add_argument('--way_test', type=int, default=5, help='way')
parser.add_argument('--num_sup_test', type=int, default=5)
parser.add_argument('--num_qur_test', type=int, default=15)
parser.add_argument('--backbone', type=str, default='conv4',help='architecture of the meta model')
parser.add_argument('--episode_train', type=int, default=240000)
parser.add_argument('--episode_test', type=int, default=600)
parser.add_argument('--start_id', type=int, default=1)
parser.add_argument('--inner_update_num', type=int, default=5)
parser.add_argument('--test_inner_update_num', type=int, default=10)
parser.add_argument('--inner_lr', type=float, default=0.01)
parser.add_argument('--outer_lr', type=float, default=0.001)
parser.add_argument('--approx', action='store_true',default=False)
parser.add_argument('--episode_batch',type=int, default=4)
parser.add_argument('--val_interval',type=int, default=2000)
parser.add_argument('--save_interval',type=int, default=2000)
#bidf-mkd
parser.add_argument('--num_sup_kd', type=int, default=30)
parser.add_argument('--num_qur_kd', type=int, default=30)
parser.add_argument('--inner_update_num_kd', type=int, default=10)
parser.add_argument('--adv', type=float, default=1.0)
parser.add_argument('--advstartit', type=int, default=-1)
#data free
parser.add_argument('--way_pretrain', type=int, default=5, help='way')
parser.add_argument('--APInum', type=int, default=100)
parser.add_argument('--pre_backbone', type=str, default='conv4', help='conv4/resnet10/resnet18')
parser.add_argument('--pretrain', action='store_true',default=False)
parser.add_argument('--generate_interval', type=int, default=200)
parser.add_argument('--generate_iterations', type=int, default=200)
parser.add_argument('--Glr', type=float, default=0.001)
#zero-order optimization
parser.add_argument('--ZO', action='store_true',default=False)
parser.add_argument('--mu', type=float, default=0.005)
parser.add_argument('--q', type=int, default=100)
parser.add_argument('--numsplit', type=int, default=5, help='parallel inference')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.multigpu

setup_seed(2022)

device=torch.device('cuda:{}'.format(args.gpu))
########
if args.dataset=='cifar100':
    img_size = 32
    args.img_size =32
    channel = 3
    args.channel=3
    class_num = 64
    args.class_num=64
elif args.dataset=='miniimagenet':
    img_size = 84
    args.img_size=84
    channel = 3
    args.channel=3
    class_num = 64
    args.class_num = 64
elif args.dataset=='omniglot':
    img_size = 28
    args.img_size = 28
    channel = 1
    args.channel = 1
    class_num = 64
    args.class_num = 64
elif args.dataset=='cub':
    img_size = 84
    args.img_size=84
    channel = 3
    args.channel=3
    class_num = 100
    args.class_num = 100
elif args.dataset=='mix':
    img_size = 84
    args.img_size = 84
    channel = 3
    args.channel = 3
    class_num = 228
    args.class_num = None
########
if args.dataset == 'mix':
    model_maml=get_model(args=args,set_maml_value=True,arbitrary_input=True)
else:
    model_maml=get_model(args,'train')
model_maml.cuda(device)

if args.dataset!='mix':
    _,_,test_loader=get_dataloader(args)
elif args.dataset=='mix':
    test_loader_cifar, test_loader_mini, test_loader_cub=get_dataloader(args)
optimizer = torch.optim.Adam(params=model_maml.parameters(), lr=args.outer_lr)
criteria = nn.CrossEntropyLoss()
maml=Maml(args)
mamlkd=MamlKD(args)
loss_all = []
acc_all=[]
max_acc_val = None
best_model_maml = None
##################################################################################################################################
timer = Timer()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
feature='{}_{}wPre_{}w_{}s_{}q_{}kds_{}kdq_{}step_{}stepkd_{}teststep_{}inner_{}outer_{}batch_{}pretrain_{}backbone_{}Ginterval_{}Giteration_{}Glr'.format(args.dataset, args.way_pretrain,args.way_train, args.num_sup_train,args.num_qur_train, args.num_sup_kd,args.num_qur_kd,args.inner_update_num,args.inner_update_num_kd,
                                                                       args.test_inner_update_num,args.inner_lr,args.outer_lr,args.episode_batch,args.pre_backbone,args.backbone,args.generate_interval,args.generate_iterations,args.Glr)


if args.approx:
    feature = feature + '_1Order'
if args.start_id!=1:
    feature = feature + '_Startfrom{}'.format(args.start_id)
if args.ZO:
    feature=feature+'_ZO{}mu{}q'.format(args.mu,args.q)
if args.APInum!=100:
    feature=feature+'_API{}'.format(args.APInum)
# save_path_prefix = './mainkd_result2'
# os.makedirs(save_path_prefix, exist_ok=True)
if (args.pre_backbone=='conv4' or args.pre_backbone=='resnet10' or args.pre_backbone=='resnet18' or args.pre_backbone=='resnet50' or args.pre_backbone=='resnet101') and args.dataset!='mix':
    pretrained_path=os.path.join(args.pretrained_path_prefix,'{}/{}/{}/{}way/model'.format(args.dataset, args.pre_backbone,'meta_train', args.way_pretrain))
    os.makedirs(pretrained_path, exist_ok=True)

##################################################################################################################################
if args.pretrain:
    pretrains(args,args.APInum,device,pretrained_path)
    print('pretrain end!')
    raise NotImplementedError
##################################################################################################################################
nz = 256
generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=channel).cuda()
transform=get_transform(args)
transform_no_toTensor=get_transform_no_toTensor(args)
if os.path.exists('./datapoolkd/' + feature):
    shutil.rmtree('./datapoolkd/' + feature)
    print('remove')
os.makedirs('./datapoolkd/' + feature,exist_ok=True)
max_batch_per_class=20
synthesizer = BlackBoxSynthesizer(args,None, None, generator,
                             nz=nz, num_classes=class_num,
                             img_size=(channel, img_size, img_size),
                             iterations=args.generate_iterations, lr_g=args.Glr,
                             synthesis_batch_size=30,
                             oh=1.0,adv=args.adv,
                             save_dir='./datapoolkd/' + feature,
                             transform=transform,transform_no_toTensor=transform_no_toTensor,
                              device=args.gpu, c_abs_list=None,max_batch_per_class=max_batch_per_class)
##################################################################################################################################
generate_idicator=True
maxAcc=None
max_acc_val=-1
max_acc_val_all=[-1,-1,-1]
max_it_all=[-1,-1,-1]
max_pm_all=[-1,-1,-1]
loss_batch, acc_batch = [], []
for task_id in range(args.start_id, args.episode_train + 1):
    if generate_idicator==True:
        if args.pre_backbone=='conv4' or args.pre_backbone=='resnet10' or args.pre_backbone=='resnet18':
            teacher = get_premodel(args).cuda(device)
            args.num_node_meta_train=args.APInum
            node_id = random.randint(0, args.num_node_meta_train - 1)
            teacher_param_specific=torch.load(os.path.join(pretrained_path,'model_specific_{}.pth'.format(node_id)))
            teacher.load_state_dict(teacher_param_specific['teacher'])
            specific=teacher_param_specific['specific']
        elif args.pre_backbone=='mix':
            if args.dataset!='mix':#sh
                random_pretrain=random.choice(['conv4','resnet10','resnet18'])
                teacher = get_premodel(args, random_pretrain).cuda(device)
                args.num_node_meta_train = args.APInum
                node_id = random.randint(0, args.num_node_meta_train - 1)
                pretrained_path = os.path.join(args.pretrained_path_prefix,'{}/{}/{}/{}way/model'.format(args.dataset, random_pretrain,'meta_train', args.way_pretrain))
                teacher_param_specific = torch.load(os.path.join(pretrained_path, 'model_specific_{}.pth'.format(node_id)))
                teacher.load_state_dict(teacher_param_specific['teacher'])
                specific = teacher_param_specific['specific']
            elif args.dataset=='mix':#mh
                random_pretrain = random.choice(['conv4', 'resnet10', 'resnet18'])
                random_dataset = random.choice(['cifar100', 'miniimagenet', 'cub'])
                if random_dataset=='cifar100':
                    args.img_size=32
                else:
                    args.img_size=84
                teacher = get_premodel(args, random_pretrain).cuda(device)
                args.num_node_meta_train = args.APInum
                node_id = random.randint(0, args.num_node_meta_train - 1)

                pretrained_path = os.path.join(args.pretrained_path_prefix,'{}/{}/{}/{}way/model'.format(random_dataset, random_pretrain,'meta_train',args.way_pretrain))
                teacher_param_specific = torch.load(
                    os.path.join(pretrained_path, 'model_specific_{}.pth'.format(node_id)))
                teacher.load_state_dict(teacher_param_specific['teacher'])
                specific = teacher_param_specific['specific']
                if random_dataset=='cifar100':
                    pass
                elif random_dataset=='miniimagenet':
                    specific=[i+64 for i in specific]
                elif random_dataset=='cub':
                    specific = [i + 128 for i in specific]
                synthesizer.transform = get_transform(args, dataset=random_dataset)
                synthesizer.transform_no_toTensor=get_transform_no_toTensor(args,dataset=random_dataset)
                transform_no_toTensor=get_transform_no_toTensor(args,dataset=random_dataset)
        # generate for support set
        synthesizer.teacher=teacher
        synthesizer.c_abs_list=specific

        timeG = Timer()
        support_tensor=synthesizer.synthesize(targets=torch.LongTensor((list(range(len(specific))))*args.num_sup_kd),student=None,mode='support',c_num=len(specific))
        #inner loop kd
        support_data=transform_no_toTensor(support_tensor)
        loss_kd = F.kl_div
        mamlkd.run_inner(model_maml=model_maml,support=support_data,criteria=loss_kd,device=device,teacher=teacher,mode='train')
        #generate for query set
        timeQ = Timer()
        if (task_id) // args.episode_batch>args.advstartit:
            query_tensor = synthesizer.synthesize(targets=torch.LongTensor((list(range(len(specific)))) * args.num_qur_kd), student=model_maml,mode='support',c_num=len(specific))#mode='query'
        else:
            query_tensor = synthesizer.synthesize(targets=torch.LongTensor((list(range(len(specific)))) * args.num_qur_kd), student=None,mode='support',c_num=len(specific))#mode='query'
        #outer loop kd
        query_data=transform_no_toTensor(query_tensor)
        query_label_relative=torch.LongTensor((list(range(len(specific))))*args.num_qur_kd).cuda(device)
        loss_outer,train_acc=mamlkd.run_outer(model_maml=model_maml,query=query_data,query_label=query_label_relative,criteria=loss_kd,device=device,teacher=teacher,mode='train')
        generate_idicator=False
    else:
        #memory
        support_data, support_label_abs, query_data, query_label_abs, specific = synthesizer.get_random_task(num_w=args.way_train,num_s=args.num_sup_train,num_q=args.num_qur_train)#SQfromS=False
        support_label = label_abs2relative(specific, support_label_abs).cuda()
        query_label = label_abs2relative(specific, query_label_abs).cuda()
        support, support_label, query, query_label = support_data.cuda(device), support_label.cuda(device), query_data.cuda(device), query_label.cuda(device)
        loss_outer, train_acc =maml.run(model_maml=model_maml,support=support,support_label=support_label,query=query,query_label=query_label,criteria=criteria,device=device,mode='train')
    loss_batch.append(loss_outer)
    acc_batch.append(train_acc)
    if task_id % args.episode_batch == 0:
        loss = torch.stack(loss_batch).sum(0)
        acc = torch.stack(acc_batch).mean()
        loss_batch, acc_batch = [], []
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if maxAcc == None or acc > maxAcc:
            maxAcc = acc
            e_count = 0
        else:
            e_count = e_count + 1
            if e_count == args.generate_interval:
                print('generate after', (task_id) // args.episode_batch)
                generate_idicator = True
                e_count = 0
                maxAcc = None
    # val
    if task_id % args.val_interval == 0:
        timeTest=Timer()
        acc_val = []
        if args.dataset!='mix':
            for test_batch in test_loader:
                data, g_label = test_batch[0].cuda(device), test_batch[1].cuda(device)
                support, support_label_relative, query, query_label_relative = data2supportquery(args, 'test', data)
                _, acc = maml.run(model_maml=model_maml, support=support, support_label=support_label_relative,
                                  query=query, query_label=query_label_relative, criteria=criteria, device=device,
                                  mode='test')
                acc_val.append(acc)
            acc_val, pm = compute_confidence_interval(acc_val)
            if acc_val > max_acc_val:
                max_acc_val = acc_val
                max_it = (task_id) // args.episode_batch
                max_pm = pm
        if args.dataset=='mix':
            test_loader_all=[test_loader_cifar,test_loader_mini,test_loader_cub]
            acc_val_all=[[],[],[]]
            for i,test_loader in enumerate(test_loader_all):
                for test_batch in test_loader:
                    data, g_label = test_batch[0].cuda(device), test_batch[1].cuda(device)
                    support, support_label_relative, query, query_label_relative = data2supportquery(args, 'test',data)
                    _, acc = maml.run(model_maml=model_maml, support=support, support_label=support_label_relative,
                                      query=query, query_label=query_label_relative, criteria=criteria,
                                      device=device,
                                      mode='test')
                    acc_val_all[i].append(acc)
                acc_val, pm = compute_confidence_interval(acc_val_all[i])
                acc_val_all[i]=acc_val
            acc_val=sum(acc_val_all)/len(acc_val_all)
            if acc_val > max_acc_val:
                max_acc_val = acc_val
                max_it = (task_id) // args.episode_batch
                max_pm = pm
            for i in range(3):
                if acc_val_all[i]>max_acc_val_all[i]:
                    max_acc_val_all[i]=acc_val_all[i]
                    max_it_all[i]=(task_id) // args.episode_batch
                    max_pm_all[i]=pm
        print((task_id) // args.episode_batch, 'test acc:', acc_val, '+-', pm)
        print(max_it, 'best test acc:', max_acc_val, '+-', max_pm)
        print('ETA:{}/{}'.format(
            timer.measure(),
            timer.measure((task_id) / (args.episode_train)))
        )