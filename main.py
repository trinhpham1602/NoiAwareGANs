from absl import app
from absl import flags
import data
import TransE as TransE_definition
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data as torch_data
from typing import Tuple
import numpy as np
import GANs
import noiAwareKGE as noiAware_difinition
from time import perf_counter
import glob
FLAGS = flags.FLAGS
flags.DEFINE_float("lr", default=0.0001, help="Learning rate value.")
flags.DEFINE_integer("seed", default=1234, help="Seed value.")
flags.DEFINE_integer("batch_size", default=512, help="Maximum batch size.")
flags.DEFINE_integer("validation_batch_size", default=64,
                     help="Maximum batch size during model validation.")
flags.DEFINE_integer("emb_dim", default=100,
                     help="Length of entity/relation vector.")
flags.DEFINE_float("margin", default=1.0,
                   help="Margin value in margin-based ranking loss.")
flags.DEFINE_integer(
    "norm", default=1, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
flags.DEFINE_integer("epochs", default=2000,
                     help="Number of training epochs.")
flags.DEFINE_string("dataset_path", default="./WN18RR",
                    help="Path to dataset.")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage.")


def main(_):
    start = perf_counter()
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
    N_triples = train_set.__len__()
    train_generator = torch_data.DataLoader(train_set, batch_size=batch_size)
    start_epoch_id = 1
    # take k% lowest h + r - t
    print("---------------------------------------------")
    print("Start the training NoiAwareGANs")
    pretrain_entities_emb = nn.Embedding(n_entities, emb_dim)
    pretrain_entities_emb.weight.data = torch.tensor(
        np.loadtxt(path + "/" + "entity2vec100.init"))
    pretrain_relations_emb = nn.Embedding(n_relations, emb_dim)
    pretrain_relations_emb.weight.data = torch.tensor(
        np.loadtxt(path + "/" + "relation2vec100.init"))
    model = noiAware_difinition.NoiAwareKGE(
        pretrain_entities_emb, pretrain_relations_emb, emb_dim, device, margin=margin)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.LogSigmoid()
    k = 0.7
    N = 1000
    all_triples_id = []
    for time in range(N):
        entities_emb = model.entities_emb.weight.data
        relations_emb = model.relations_emb.weight.data
        hrt_embs = torch.zeros((N_triples, 3, emb_dim), dtype=float)
        norm_order = []
        #triples_id = []
        # in train_set, lines is random and diference with origin data set
        for i in range(N_triples):
            (h_id, r_id, t_id) = train_set.__getitem__(i)
            all_triples_id.append([h_id, r_id, t_id])
            h_emb = entities_emb[h_id]
            r_emb = relations_emb[r_id]
            t_emb = entities_emb[t_id]
            norm = torch.norm(h_emb+r_emb-t_emb, p=1)
            norm_order.append((norm.item(), i))
            #triples_id.append([h_id, r_id, t_id])
            hrt_embs[i][0] = h_emb
            hrt_embs[i][1] = r_emb
            hrt_embs[i][2] = t_emb

        dtype = [("norm", float), ("order", int)]
        norm_order = np.array(norm_order, dtype=dtype)
        norm_order = np.sort(norm_order, order="norm")
        k_percent_lowest = torch.zeros((int(k*N_triples), 3, emb_dim))
        for i in range(int(k*N_triples)):
            k_percent_lowest[i] = hrt_embs[norm_order[i][1]]
        k_percent_lowest = k_percent_lowest.to(device).float()
        # define GANs
        epochs4GANs = 1
        D, G = GANs.run(k_percent_lowest, emb_dim,
                        learning_rate, batch_size, epochs4GANs)
        # train noiAwareKGE
        for _ in range(start_epoch_id, epochs + 1):
            model.train()
            for local_heads, local_relations, local_tails in train_generator:
                local_heads, local_relations, local_tails = (local_heads.to(device), local_relations.to(device),
                                                             local_tails.to(device))
                positive_triples = torch.stack(
                    (local_heads, local_relations, local_tails), dim=1).long()
                head_or_tail = torch.randint(
                    high=2, size=local_heads.size(), device=device)
                random_entities = torch.randint(
                    high=len(entity2id), size=local_heads.size(), device=device)
                broken_heads = torch.where(
                    head_or_tail == 1, random_entities, local_heads)
                broken_tails = torch.where(
                    head_or_tail == 0, random_entities, local_tails)
                negative_triples = torch.stack(
                    (broken_heads, local_relations, broken_tails), dim=1).long()
                optimizer.zero_grad()
                loss = model(positive_triples, negative_triples, D, G)
                loss = criterion(loss)
                loss.mean().backward()

                optimizer.step()
        print("Finished interator: ", time + 1)
    end = perf_counter()
    print("The NoiAwareGAN is trained")
    print("total time pretrain and train NoiAwareGANs is ", end - start)
    print("---------------------------------------------")
    entities_emb = model.entities_emb.weight.data.cpu().numpy()
    relations_emb = model.relations_emb.weight.data.cpu().numpy()
    directory = "output"
    path = os.path.join("./", directory)
    if not os.path.exists(path):
        os.mkdir(path)
    f = open("./output/entity2id.txt", "w")
    np.savetxt("./output/entities_emb.txt", entities_emb)
    np.savetxt("./output/relations_emb.txt", relations_emb)
    print("Done!")


if __name__ == '__main__':
    app.run(main)
