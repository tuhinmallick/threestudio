import torch
import torch.nn.functional as F


def generator_loss(discriminator, inputs, reconstructions, cond=None):
    if cond is None:
        logits_fake = discriminator(reconstructions.contiguous())
    else:
        logits_fake = discriminator(
            torch.cat((reconstructions.contiguous(), cond), dim=1)
        )
    return -torch.mean(logits_fake)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def discriminator_loss(discriminator, inputs, reconstructions, cond=None):
    if cond is None:
        logits_real = discriminator(inputs.contiguous().detach())
        logits_fake = discriminator(reconstructions.contiguous().detach())
    else:
        logits_real = discriminator(
            torch.cat((inputs.contiguous().detach(), cond), dim=1)
        )
        logits_fake = discriminator(
            torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
        )
    return hinge_d_loss(logits_real, logits_fake).mean()
