import torch


def feature_matching_loss(fmap_real, fmap_fake):
    loss = 0
    for d_real, d_fake in zip(fmap_real, fmap_fake):
        for o_real, o_fake in zip(d_real, d_fake):
            loss += torch.mean(torch.abs(o_real - o_fake))
    return loss


def discriminator_loss(disc_real, disc_fake, scale=1.0):
    loss = 0
    for i, (d_real, d_fake) in enumerate(zip(disc_real, disc_fake)):
        real_loss = torch.mean((1 - d_real) ** 2)
        fake_loss = torch.mean(d_fake**2)
        if i < len(disc_real):
            loss += real_loss + fake_loss
        else:
            loss += scale * (real_loss + fake_loss)
    return loss


def generator_loss(disc_outputs):
    loss = 0
    for d_fake in disc_outputs:
        d_fake = d_fake.float()
        loss += torch.mean((1 - d_fake) ** 2)
    return loss
