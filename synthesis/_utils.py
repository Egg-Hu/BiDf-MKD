import torch
import numpy as np
from PIL import Image
import os, random, math
import sys


sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from black_box_tool import get_84_transform

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
    
def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images( imgs, col=col ).transpose( 1, 2, 0 ).squeeze()
        imgs = Image.fromarray( imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        #output_filename = output.strip('.png')
        output_filename = output[:-4]
        for idx, img in enumerate(imgs):
            img = Image.fromarray( img.transpose(1, 2, 0).squeeze() )
            img.save(output_filename+'-%d.png'%(idx))

def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2) # make it channel first
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))
    
    pack = np.zeros( (C, H*row+padding*(row-1), W*col+padding*(col-1)), dtype=images.dtype )
    for idx, img in enumerate(images):
        h = (idx // col) * (H+padding)
        w = (idx % col) * (W+padding)
        pack[:, h:h+H, w:w+W] = img
    return pack


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
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




class ImagePool2(object):
    def __init__(self,args, root,num_classes,transform,max_batch_per_class):
        self.args=args
        self.root_support=os.path.join(root,'support')
        self.root_query = os.path.join(root, 'query')
        #support pool
        for c_abs in range(num_classes):
            os.makedirs(os.path.join(self.root_support,str(c_abs)), exist_ok=True)
            os.makedirs(os.path.join(self.root_query, str(c_abs)), exist_ok=True)
        self._idx = [0]*num_classes
        self.max_batch_per_class=max_batch_per_class
        self.ready_class=[]
        self.transform=transform

    def add(self, imgs, c_abs_list=None,synthesis_batch_size_per_class=None,mode=None):
        c_abs_targets=torch.LongTensor(c_abs_list*synthesis_batch_size_per_class)
        for c_abs in c_abs_list:
            if mode=='support':
                root=os.path.join(self.root_support,str(c_abs))
            if mode=='query':
                root=os.path.join(self.root_query,str(c_abs))
            imgs_c=imgs[c_abs_targets==c_abs]
            save_image_batch(imgs_c, os.path.join( root, "%d.png"%(self._idx[c_abs]) ), pack=False)
            self._idx[c_abs]+=1
            self._idx[c_abs]=self._idx[c_abs]%self.max_batch_per_class
            if c_abs not in self.ready_class:
                self.ready_class.append(c_abs)

    def get_random_task(self,num_w,num_s,num_q):
        select_way=random.sample(self.ready_class,num_w)
        support_x=[]
        query_x=[]
        support_y_abs=[]
        query_y_abs = []
        for c_relative,c_abs in enumerate(select_way):
            c_abs_path = os.path.join(self.root_support, str(c_abs))
            image_name_list = os.listdir(c_abs_path)
            image_name_list = random.sample(image_name_list, (num_s+num_q))
            if self.args.dataset!='mix':
                if self.args.dataset == 'omniglot':
                    select_image = [self.transform(Image.open(os.path.join(c_abs_path, image_name)).convert('L')) for image_name in image_name_list]
                else:
                    select_image = [self.transform(Image.open(os.path.join(c_abs_path, image_name)).convert('RGB')) for image_name in image_name_list]
            elif self.args.dataset=='mix':
                if c_abs<=63:
                    transform=get_84_transform(self.args, dataset='cifar100')
                else:
                    transform=get_84_transform(self.args, dataset='cub')
                select_image = [transform(Image.open(os.path.join(c_abs_path, image_name)).convert('RGB')) for
                                image_name in image_name_list]

            select_image = torch.stack(select_image, dim=0)
            support_x.append(select_image[:num_s])
            query_x.append(select_image[num_s:(num_s + num_q)])
            support_y_abs.append(torch.LongTensor([c_abs] * num_s))
            query_y_abs.append(torch.LongTensor([c_abs] * num_q))
        return torch.cat(support_x,dim=0),torch.cat(support_y_abs,dim=0),torch.cat(query_x,dim=0),torch.cat(query_y_abs,dim=0),select_way
