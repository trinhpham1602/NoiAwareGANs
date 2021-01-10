import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class Discriminator(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(emb_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, hrt_concat_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(hrt_concat_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        return self.gen(x)


def run(postriples, emb_dim, lr, epochs, negative_sample_size, n_neg):

    disc = Discriminator(emb_dim).to(device)
    gen = Generator(3*emb_dim).to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for _ in range(epochs):

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(n_neg, 3*emb_dim).to(
            device)
        fake_probab = gen(noise).view(-1)
        true_neg_trips = noise[torch.topk(
            fake_probab, negative_sample_size).indices]
        fake_trips = noise.reshape((n_neg, 3, emb_dim))
        disc_real = disc(postriples).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake_trips).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        true_neg_trips = true_neg_trips.reshape(
            (len(true_neg_trips), 3, emb_dim))
        output = disc(fake_trips).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
    return disc, gen
