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
flags.DEFINE_integer("epochs", default=1000,
                     help="Number of training epochs.")
flags.DEFINE_string("dataset_path", default="./FB15k-237",
                    help="Path to dataset.")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage.")


def take_true_neg_trips(G: GANs.Generator, pos: torch.LongTensor, model: NoiAware_definition.NoiAware, n_neg, n_entities, emb_dim, sizeof_true_negs, device):
    # h r t'
    negs = pos.repeat((n_neg, 1))
    break_head = random.sample(range(n_entities), n_neg + 1)
    if pos[0] in break_head:
        break_head.remove(pos[0])
    else:
        break_head.pop()
    break_tail = random.sample(range(n_entities), n_neg + 1)
    if pos[2] in break_tail:
        break_tail.remove(pos[2])
    else:
        break_tail.pop()
    break_head = torch.tensor(break_head).to(device)
    break_tail = torch.tensor(break_tail).to(device)

    negs[:, 2] = break_tail
    negs_emb = model._get_emb(negs)
    # concat hrt in negs
    negs_emb = negs_emb.reshape(len(negs_emb), 1, emb_dim*3)
    true_negs_tail = negs[torch.topk(
        G.forward(negs_emb).view(-1), int(sizeof_true_negs/2)).indices]
    # h' r t
    negs = pos.repeat((n_neg, 1))

    negs[:, 0] = break_head
    negs_emb = model._get_emb(negs)
    negs_emb = negs_emb.reshape(len(negs_emb), 1, emb_dim*3)
    true_negs_head = negs[torch.topk(
        G.forward(negs_emb).view(-1), int(sizeof_true_negs/2)).indices]
    return torch.cat((true_negs_head, true_negs_tail)).to(device)


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

    pretrain_entities_emb = nn.Embedding(n_entities, emb_dim)
    pretrain_entities_emb.weight.data = torch.tensor(
        np.load(path + "/" + "entity_embedding.npy"))
    pretrain_relations_emb = nn.Embedding(n_relations, emb_dim)
    pretrain_relations_emb.weight.data = torch.tensor(
        np.load(path + "/" + "relation_embedding.npy"))
    print("Finished load pretrain embedding vectors")
    model = NoiAware_definition.NoiAware(
        pretrain_entities_emb, pretrain_relations_emb, n_entities, n_relations, emb_dim, device, margin=margin)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.LogSigmoid()
    #
    epochs4GAN = 100
    negative_sample_size = 1024
    n_negs = int(n_entities/2)
    k = int(batch_size*0.7)
    start = perf_counter()
    directory = "output"
    path = os.path.join("./", directory)
    if not os.path.exists(path):
        os.mkdir(path)
    for epoch in range(1, epochs + 1):
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
                            epochs4GAN, negative_sample_size, n_negs)
            # tao negative_triples
            blocks_true_negs_each_pos = [take_true_neg_trips(
                G, pos, model, n_negs, n_entities, emb_dim, negative_sample_size, device) for pos in positive_triples]
            optimizer.zero_grad()
            loss = model(positive_triples, blocks_true_negs_each_pos,
                         negative_sample_size, D)
            loss = criterion(loss)
            loss.mean().backward()
            optimizer.step()
        if epoch % 50 == 0:
            entities_emb = model.entities_emb.weight.data.cpu().numpy()
            relations_emb = model.relations_emb.weight.data.cpu().numpy()
            np.savetxt("./output/entities_emb.txt", entities_emb)
            np.savetxt("./output/relations_emb.txt", relations_emb)
    end = perf_counter()
    print("Done with: ", end - start)


if __name__ == '__main__':
    app.run(main)
