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
import time
import datatime
import sys
import os
import tqdm
from config import opt

if __name__ == "__main__":
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
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    prev_time = time.time()
    for epoch in range(opt.epochs):
        for i, batch in tqdm.tqdm(enumerate(train_loader)):
            real_A, real_B = [item.to(device) for item in batch]
            # Adversarial ground truths
            valid = Variable(np.ones((real_A.size(0), *D_A.output_shape)).floats(), requires_grad=False)
            fake = Variable(np.zeros((real_A.size(0), *D_A.output_shape)).floats(), requires_grad=False)
            G_AB.train()
            G_BA.train()

            optimize_G.zero_grad()
            fake_B = G_BA(real_A)
            fake_A = G_AB(real_B)
            loss_A = critrion_identify(fake_A, real_A)
            loss_B = critrion_identify(fake_B, real_B)

            loss_identify = (loss_A + loss_B)/2
            loss_AB = critrion_gan(D_B(fake_B),valid)
            loss_BA = critrion_gan(D_A(fake_A),valid)
            loss_GAN = (loss_AB+loss_BA)/2
            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)
            loss_cycle_A = critrion_cycle(recov_A , real_A)
            loss_cycle_B = critrion_cycle(recov_B , real_B)
            loss_cycle = (loss_cycle_A+loss_cycle_B)/2
            total_loss = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identify

            total_loss.backward()
            optimize_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimize_D_A.zero_grad()

            # Real loss
            loss_real = critrion_gan(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = critrion_gan(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimize_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimize_D_B.zero_grad()

            # Real loss
            loss_real = critrion_gan(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = critrion_gan(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimize_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2
            # Print log
            # Determine approximate time left
            batches_done = epoch * len(train_loader) + i
            batches_left = opt.n_epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(train_loader),
                    loss_D.item(),
                    loss_B.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identify.item(),
                    time_left,
                )
            )

        # Update learning rates
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
            torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))














