import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import NoiAware

import numpy as np
import random
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


def run(model: NoiAware, pos_triples, entities, emb_dim, lr, step, n_negs, k_negs):
    disc = Discriminator(emb_dim).to(device)  # environment
    gen = Generator(3*emb_dim).to(device)  # agent
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    b = 0
    for _ in range(step):
        G_d = 0
        G_g = 0
        r_sum = 0
        opt_disc.zero_grad()
        opt_gen.zero_grad()
        global high_quality_negs
        high_quality_negs = []
        for pos in pos_triples:
            true_entities = np.array([pos[0], pos[2]])
            ignore_true_entities = np.setdiff1d(entities, true_entities)
            head_or_tail = np.random.randint(2, size=n_negs)
            rand_entities = random.sample(list(ignore_true_entities), n_negs)
            neg_triples = []
            for i, val in enumerate(head_or_tail):
                if val == 1:
                    neg_triples.append([pos[0], pos[1], rand_entities[i]])
                else:
                    neg_triples.append([rand_entities[i], pos[1], pos[2]])
            # get embedding

            pos_emb = model._get_emb4triple(pos.to(device)).detach()
            pos_emb[2] = -pos_emb[2]
            sum_hrt = torch.sum(pos_emb, dim=1)
            neg_triples = torch.tensor(neg_triples)
            neg_embs = model._get_emb(neg_triples.to(device)).detach()
            temp_neg_embs = neg_embs
            temp_neg_embs[2] = -temp_neg_embs[2]
            sum_neg_hrts = torch.sum(temp_neg_embs, dim=1)
            G_d += -torch.log(disc(sum_hrt)) - \
                torch.sum(torch.log(1 - disc(sum_neg_hrts)))
            reward = - disc(sum_neg_hrts).detach()
            r_sum += reward
            G_g += (reward - b) * \
                torch.log(gen(neg_embs.view(n_negs, 3*emb_dim)))
            prob_negs = gen(neg_embs.view(n_negs, 3*emb_dim)).detach().view(-1)
            high_quality_negs.append(
                neg_triples[torch.topk(prob_negs, k_negs).indices])
        G_d.backward()
        opt_disc.step()
        G_g.sum().backward()
        opt_gen.step()
        b = r_sum/pos_triples.size(0)
    # choose k high quality negs
    test_negs = model._get_emb(
        high_quality_negs[0].to(device)).view(k_negs, 3*emb_dim)
    print(gen(test_negs))

    return disc, gen
