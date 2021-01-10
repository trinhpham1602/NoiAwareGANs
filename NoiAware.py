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
        positive_triples = self._get_emb(positive_triples).float()
        positive_triples[:, 2] = -positive_triples[:, 2]
        input_disc = torch.sum(positive_triples, dim=1)  # vector: h + r - t
        confident_scores = D.forward(input_disc)
        sum_scores = 0.0
        for inx, hrt in enumerate(positive_triples):
            _4Apos = - \
                torch.log(torch.sigmoid(self.margin - self._distance(hrt)))
            true_negs = block_of_negative_triples[inx]
            _4negs = 1/negative_sample_size * \
                torch.sum(torch.log(torch.sigmoid(
                    self.margin - self._distance(true_negs))))
            withConfScore = confident_scores[inx] * (_4Apos + _4negs)
            sum_scores = sum_scores + withConfScore

        return sum_scores

    def _get_emb(self, triplets):
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]

        return torch.stack((self.entities_emb(heads), self.relations_emb(
            relations), self.entities_emb(tails)), dim=1)  # size: batch_size x 3 x emb_dim

    def predict(self, triplets: torch.LongTensor):
        return self._distance(triplets)

    def _distance(self, triplets):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        assert triplets.size()[1] == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return (self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)).norm(p=self.norm,
                                                                                                          dim=1)
