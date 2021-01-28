import numpy as np
import torch
import torch.nn as nn
import os
import GANs
from torch.utils import data as torch_data


class NoiAware(nn.Module):

    def __init__(self, pretrain_entities_emb: nn.Embedding, pretrain_relations_emb: nn.Embedding, entity_count, relation_count, emb_dim, device, norm=1, margin=24):
        super(NoiAware, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.norm = norm
        self.margin = margin
        self.emb_dim = emb_dim
        self.entities_emb = pretrain_entities_emb
        self.relations_emb = pretrain_relations_emb

    def forward(self, positive_triples, block_of_negative_triples, negative_sample_size, D: GANs.Discriminator):
        # G take hrt concat
        positive_embs = self._get_emb(positive_triples)
        positive_embs[:, 2] = -positive_embs[:, 2]
        input_disc = torch.sum(positive_embs, dim=1)  # vector: h + r - t
        confident_scores = D.forward(input_disc).view(-1).to(self.device)
        pos_scores = - \
            torch.log(torch.sigmoid(self.margin -
                                    self._distance(positive_triples))).to(self.device)
        neg_scores = torch.tensor([torch.sum(1/negative_sample_size*torch.log(torch.sigmoid(
            self.margin - self._distance(neg_trips)))) for neg_trips in block_of_negative_triples]).to(self.device)
        sum_scores = confident_scores*(pos_scores + neg_scores)
        return sum_scores

    def _get_emb(self, triplets):
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]

        return torch.stack((self.entities_emb(heads), self.relations_emb(
            relations), self.entities_emb(tails)), dim=1)  # size: batch_size x 3 x emb_dim

    def _get_emb4triple(self, triple):
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        return torch.stack((self.entities_emb(head), self.relations_emb(
            relation), self.entities_emb(tail)), dim=1)

    def predict(self, triplets: torch.LongTensor):
        return self._distance(triplets)

    def _distance(self, triplets):
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return (self.entities_emb(heads).to(self.device) + self.relations_emb(relations).to(self.device) - self.entities_emb(tails).to(self.device)).norm(p=self.norm,
                                                                                                                                                          dim=1).to(self.device)
