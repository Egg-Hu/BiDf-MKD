import sys
import os

from torch.nn import CrossEntropyLoss
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseSynthesis
from ._utils import ImagePool2


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class BlackBoxSynthesizer(BaseSynthesis):
    def __init__(self, args,teacher, student, generator, nz, num_classes, img_size,
                 iterations=200, lr_g=0.1,
                 synthesis_batch_size=128,
                  oh=1,adv=1,
                 save_dir='run/cmi', transform=None,transform_no_toTensor=None,
                 device='cpu', c_abs_list=None,max_batch_per_class=20):
        super(BlackBoxSynthesizer, self).__init__(teacher, student)
        self.args=args
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.oh = oh
        self.adv=adv
        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.c_abs_list=c_abs_list
        self.transform = transform
        self.data_pool = ImagePool2(args=self.args,root=self.save_dir, num_classes=self.num_classes, transform=self.transform,max_batch_per_class=max_batch_per_class)
        self.generator = generator.to(device).train()
        self.device = device
        self.hooks = []
        self.transform_no_toTensor=transform_no_toTensor
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)


    def synthesize(self, targets=None,student=None,mode=None,c_num=5,support=None):
        self.synthesis_batch_size =len(targets)//c_num
        self.hooks = []
        self.teacher.eval()
        ########################
        best_cost = 1e6

        z = torch.randn(size=(len(targets), self.nz), device=self.device).requires_grad_()
        best_inputs = self.generator(z).data
        targets = torch.LongTensor(targets).to(self.device)
        reset_model(self.generator)
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g, betas=[0.5, 0.999])
        criteria_adv=F.kl_div
        for it in range(self.iterations):
            inputs = self.generator(z)
            inputs_change=self.transform_no_toTensor(inputs)
            #############################################
            #Loss
            #############################################
            if self.args.ZO==False:
                t_out = self.teacher(inputs_change)
                loss_oh = F.cross_entropy( t_out, targets )
                loss = self.oh * loss_oh
                if student !=None:
                    with torch.no_grad():
                        s_out = student(inputs_change)
                    mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                    loss_adv = -(criteria_adv(F.softmax(t_out,dim=-1),F.softmax(s_out.detach(),dim=-1), reduction='none').sum(1) * mask).mean()
                    loss=loss+self.adv*loss_adv
            else:
                inputs_change.requires_grad_(True)
                inputs_change.retain_grad()
                #zero-order black-box
                criterion = CrossEntropyLoss(reduction='none').cuda(self.device)
                with torch.no_grad():
                    m, sigma = 0, 100  # mean and standard deviation
                    mu = torch.tensor(self.args.mu).cuda(self.device)
                    q = torch.tensor(self.args.q).cuda(self.device)
                    batch_size=inputs_change.size()[0]
                    channel = inputs_change.size()[1]
                    h = inputs_change.size()[2]
                    w = inputs_change.size()[3]
                    d = channel * h * w
                    #original
                    original_t_out=self.teacher(inputs_change)
                    original_loss_oh=criterion( original_t_out, targets )
                    if student!=None:
                        with torch.no_grad():
                            s_out = student(inputs_change)
                        mask = (s_out.max(1)[1] == original_t_out.max(1)[1]).float()
                        original_loss_adv=-(F.l1_loss(original_t_out,s_out.detach() , reduction='none').sum(1) * mask)
                        assert len(original_loss_adv.shape)==1,'error'
                    #parallel query
                    num_split = self.args.numsplit
                    grad_est = torch.zeros((self.args.q//num_split)*batch_size, d).cuda(self.device)
                    new_original_loss_oh = original_loss_oh.repeat((self.args.q // num_split))
                    if student != None:
                        new_original_loss_adv = original_loss_adv.repeat((self.args.q // num_split))
                        assert len(new_original_loss_adv.shape) == 1, 'error'
                        new_s_out = s_out.repeat((self.args.q // num_split), 1)
                    assert len(new_original_loss_oh.shape) == 1, 'error'
                    new_targets = targets.repeat((self.args.q // num_split))
                    inputs_change_flatten = torch.flatten(inputs_change, start_dim=1).repeat((self.args.q // num_split),1).cuda(self.device)

                    for k in range(num_split):
                        u = torch.normal(m, sigma, size=((self.args.q//num_split)*batch_size, d))
                        u_norm = torch.norm(u, p=2, dim=1).reshape((self.args.q//num_split)*batch_size, 1).expand((self.args.q//num_split)*batch_size, d)  # dim -- careful
                        u = torch.div(u, u_norm).cuda()  # (q*batch_size, d)
                        # forward difference
                        inputs_change_flatten_q = inputs_change_flatten + mu * u
                        inputs_change_q = inputs_change_flatten_q.view((self.args.q//num_split)*batch_size, channel, h, w)
                        t_out_q = self.teacher(inputs_change_q)
                        loss_oh_q = criterion(t_out_q, new_targets)
                        loss_diff = (loss_oh_q - new_original_loss_oh)#torch.tensor
                        assert len(loss_diff.shape) == 1, 'error'
                        if student != None:
                            mask = (new_s_out.max(1)[1] == t_out_q.max(1)[1]).float()
                            loss_adv_q = -(F.l1_loss(t_out_q, new_s_out.detach(), reduction='none').sum(1) * mask)
                            assert len(loss_adv_q.shape)==1
                            loss_diff = loss_diff + self.adv * (loss_adv_q - new_original_loss_adv)#torch.tensor
                        assert loss_diff.shape[0]==(self.args.q//num_split)*batch_size
                        assert loss_diff.shape[0] == (self.args.q//num_split)*batch_size
                        grad_est =grad_est+ (d / q) * u * loss_diff.reshape((self.args.q//num_split)*batch_size, 1).expand_as(u) / mu
                        assert grad_est.shape[0]==(self.args.q//num_split)*batch_size
                    grad_est=grad_est.reshape((self.args.q//num_split),batch_size,d).sum(0)
                    assert grad_est.shape[0] ==  batch_size

                inputs_change_flatten = torch.flatten(inputs_change, start_dim=1).cuda(self.device)
                grad_est_no_grad = grad_est.detach()
                loss = torch.sum(inputs_change_flatten * grad_est_no_grad, dim=-1).mean()

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # save best inputs and reset data iter
        self.data_pool.add( imgs=best_inputs ,c_abs_list=self.c_abs_list,synthesis_batch_size_per_class=self.synthesis_batch_size,mode=mode)
        return best_inputs

    def get_random_task(self, num_w=5, num_s=5, num_q=15):
        return self.data_pool.get_random_task(num_w=num_w, num_s=num_s, num_q=num_q)