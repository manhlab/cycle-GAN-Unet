import torch.optim as optim
import torch

from dataloader import CycleganDataset
from .utils import *
from .model import *

from torchvision import dataset
import torch.utils.functional as F
from torch.utils.data import Dataset, Dataloader
import torch.nn as nn
import torch
from torch.autograd import Variable


if __name__ == "__main__":
    opt = []
    # Initialize generator and discriminator
    input_shape = (opt.channels, opt.img_height, opt.img_width)
    G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)
    critrion_gan, critrion_cycle, critrion_identify = get_loss()
    transform_ = get_transform(opt)
    optimize_G, optimize_D_A, optimize_D_B = get_optimizer()
    scheduler_G, scheduler_D_A, scheduler_D_B = get_scheduler(
        opt, optimize_G, optimize_D_A, optimize_D_B
    )
    device = get_device()
    if device=="cude":
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        critrion_gan.cuda()
        critrion_cycle.cuda()
        critrion_identify.cuda()

    train_loader = Dataloader(
        CycleganDataset("../data/train", transform_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    valid_loader = Dataloader(
        CycleganDataset("../data/valid", transform_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    # ----------
    #  Training
    # ----------
    prev_time = time.time()
    for epoch in range(opt.epochs):
        for i, batch in enumerate(train_loader):
            real_A, real_B = [item.to(device) for item in batch]
            # Adversarial ground truths
            valid = Variable(np.ones((real_A.size(0), *D_A.output_shape)).floats(), requires_grad=False)
            fake = Variable(np.zeros((real_A.size(0), *D_A.output_shape)).floats(), requires_grad=False)




