from absl import app
from absl import flags
import data
import NoiAware as NoiAware_definition
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data as torch_data
from typing import Tuple
import numpy as np
import GANs
from time import perf_counter
import glob
import random
FLAGS = flags.FLAGS
flags.DEFINE_float("lr", default=0.0001, help="Learning rate value.")
flags.DEFINE_integer("seed", default=1234, help="Seed value.")
flags.DEFINE_integer("batch_size", default=1024, help="Maximum batch size.")
flags.DEFINE_integer("validation_batch_size", default=64,
                     help="Maximum batch size during model validation.")
flags.DEFINE_integer("emb_dim", default=500,
                     help="Length of entity/relation vector.")
flags.DEFINE_float("margin", default=24.0,
                   help="Margin value in margin-based ranking loss.")
flags.DEFINE_integer(
    "norm", default=1, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
flags.DEFINE_integer("epochs", default=1,
                     help="Number of training epochs.")
flags.DEFINE_string("dataset_path", default="./FB15k-237",
                    help="Path to dataset.")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage.")


def main(_):

    torch.random.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    path = FLAGS.dataset_path
    train_file = open(path + "/" + "train.txt", "rb")
    valid_file = open(path + "/" + "valid.txt", "rb")
    test_file = open(path + "/" + "test.txt", "rb")
    with open(path + "/" + "all_triples.txt", "wb") as outfile:
        outfile.write(train_file.read())
        outfile.write(valid_file.read())
        outfile.write(test_file.read())

    train_path = os.path.join(path, "all_triples.txt")

    entity2id, relation2id = data.create_mappings(train_path)
    n_entities = len(entity2id)
    n_relations = len(relation2id)

    batch_size = FLAGS.batch_size
    emb_dim = FLAGS.emb_dim
    margin = FLAGS.margin
    norm = FLAGS.norm
    learning_rate = FLAGS.lr
    epochs = FLAGS.epochs
    device = torch.device('cuda') if FLAGS.use_gpu else torch.device('cpu')

    train_set = data.KGDataset(train_path, entity2id, relation2id)
    train_generator = torch_data.DataLoader(train_set, batch_size=batch_size)
    print("Load pretrain embedding vectors")
    pretrain_entities_emb = nn.Embedding(n_entities, emb_dim)
    pretrain_entities_emb.weight.data = torch.tensor(
        np.load(path + "/" + "entity_embedding.npy"))
    pretrain_relations_emb = nn.Embedding(n_relations, emb_dim)
    pretrain_relations_emb.weight.data = torch.tensor(
        np.load(path + "/" + "relation_embedding.npy"))

    model = NoiAware_definition.NoiAware(
        pretrain_entities_emb, pretrain_relations_emb, n_entities, n_relations, emb_dim, device, margin=margin)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.LogSigmoid()
    #
    epochs4GAN = 1
    negative_sample_size = 1024
    n_negs_rand = int(n_entities/2)
    k = int(batch_size*0.7)
    for _ in range(1, epochs + 1):
        model.train()
        for local_heads, local_relations, local_tails in train_generator:
            local_heads, local_relations, local_tails = (local_heads.to(
                device), local_relations.to(device), local_tails.to(device))
            positive_triples = torch.stack(
                (local_heads, local_relations, local_tails), dim=1).long()
            pos_embs = model._get_emb(positive_triples)
            # make -t
            pos_embs[:, 2] = -pos_embs[:, 2]
            # take k%

            plus_hrt = torch.sum(pos_embs, dim=1)
            distance_hrt = torch.norm(plus_hrt, dim=1)
            pos_of_k_percent = plus_hrt[torch.topk(distance_hrt, k).indices]

            D, G = GANs.run(pos_of_k_percent, emb_dim, learning_rate,
                            epochs4GAN, negative_sample_size, n_negs_rand)
            # tao negative_triples

            negative_triples = []
            for [h, r, t] in positive_triples:
                h = h.item()
                r = r.item()
                t = t.item()
                h_or_t = torch.randint(
                    high=2, size=(n_negs_rand,), device=device)
                rand_neg_samples = random.sample(
                    range(n_entities), n_negs_rand + 1)
                for inx, ber in enumerate(h_or_t):
                    if ber == 1:
                        break_head = rand_neg_samples[inx]
                        if break_head == h:
                            break_head = rand_neg_samples[inx + 1]
                        negative_triples.append([break_head, r, t])
                    else:
                        break_tail = rand_neg_samples[inx]
                        if break_tail == t:
                            break_tail = rand_neg_samples[inx + 1]
                        negative_triples.append([h, r, break_tail])
            negative_triples = torch.tensor(negative_triples).to(device)
            print(negative_triples.size())
            # dung GAN de loc negative sample
            negative_blocks = torch_data.DataLoader(
                negative_triples, n_negs_rand)
            # return len: batch_size, negative_size for each block
            blocks_true_negs_each_pos = []
            for negs in negative_blocks:
                negs_embs = model._get_emb(negs)
                concat_hrt = torch.reshape(
                    negs_embs, (len(negs_embs), 1, emb_dim * 3))
                probabs_neg = G.forward(concat_hrt).view(-1)
                true_negs = negs[torch.topk(
                    probabs_neg, negative_sample_size).indices]
                blocks_true_negs_each_pos.append(true_negs)
            print("alo tai day")
            optimizer.zero_grad()
            loss = model(positive_triples, blocks_true_negs_each_pos,
                         negative_sample_size, D)
            print("alo tai day")
            loss = criterion(loss)
            loss.mean().backward()
            optimizer.step()
    print("Done")


if __name__ == '__main__':
    app.run(main)
