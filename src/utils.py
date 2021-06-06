import random
import torch
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )


def get_loss():
    critrion_gan = torch.nn.MSELoss()
    critrion_cycle = torch.nn.L1Loss()
    critrion_identify = torch.nn.L1Loss()
    return critrion_gan, critrion_cycle, critrion_identify


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_transform(opt):
    # Image transformations
    transforms_ = [
        transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((opt.img_height, opt.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms_


def get_optimizer(opt):
    optimize_G = torch.optim.Adam(opt.lr)
    optimize_D_A = torch.optim.Adam(opt.lr)
    optimize_D_B = torch.optim.Adam(opt.lr)
    return optimize_G, optimize_D_A, optimize_D_B


def get_scheduler(opt, optimize_G, optimize_D_A, optimize_D_B):
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimize_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimize_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimize_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    return scheduler_G, scheduler_D_A, scheduler_D_B

