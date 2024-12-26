import os
import time
import shutil
import math
import scipy.io as sio    
import torch
import numpy as np
from torch.optim import SGD, Adam
from skimage import io
#from tensorboardX import SummaryWriter
import pytorch_ssim


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=False):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
#    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log#, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb

def calc_psnr_ssim(sr, hr, dataset=None, scale=1, rgb_range=1):
    psnr = calc_psnr(sr, hr, dataset, scale, rgb_range)
    ssim = calc_ssim(sr, hr, dataset, scale, rgb_range)
    return psnr, ssim
    
def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


def calc_ssim(sr, hr, dataset=None, scale=1, rgb_range=1):
    sr = sr/rgb_range
    hr = hr/rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if sr.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = sr.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                sr = sr.mul(convert).sum(dim=1).unsqueeze(1)
                hr = hr.mul(convert).sum(dim=1).unsqueeze(1)
        else:
            shave = scale + 6

    sr = sr[..., shave:-shave, shave:-shave]
    hr = hr[..., shave:-shave, shave:-shave]
    ssim = pytorch_ssim.ssim(sr.detach().cpu(),hr.detach().cpu())
    return ssim

def save_results(filename, iterm_save):
            tensor_cpu = (iterm_save*255).byte().permute(1, 2, 0).cpu()
            io.imsave(('{}.png'.format(filename)), tensor_cpu) 
            
def save_results_mat(filename, iterm_save):
            tensor_cpu = iterm_save.permute(1, 2, 0).cpu().numpy()
            sio.savemat(('{}.mat'.format(filename)), {'sr':tensor_cpu}) 
                
def save_psnr(save_path, psnr_all):
    sio.savemat(('{}.mat'.format(save_path)), {'psnr_all':psnr_all}) 

def create_gaussian_kernel(kernel_size: int, sigma: float):
    # 检查kernel_size是否为奇数
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    
    # 生成中心坐标
    center = (kernel_size - 1) / 2
    
    # 生成x和y坐标网格
    x, y = torch.meshgrid(torch.arange(kernel_size, dtype=torch.float32) - center,
                          torch.arange(kernel_size, dtype=torch.float32) - center,
                          indexing='ij')
    
    # 计算高斯核
    kernel = gaussian(x, sigma) * gaussian(y, sigma)
    
    # 归一化核
    kernel = kernel / kernel.sum()
    
    # 将核转换为4维张量 (1, 1, kernel_size, kernel_size)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    return  kernel

def gaussian(x, sigma):
    return torch.exp(-x**2 / (2 * sigma**2)) / (math.sqrt(2 * math.pi) * sigma)

def normalized(X):
    maxX = np.max(X)
    minX = np.min(X)
    X = (X-minX)/(maxX - minX)
    return X

def setRange(X, maxX = 1, minX = 0):
    X = (X-minX)/(maxX - minX+0.001)
    return X