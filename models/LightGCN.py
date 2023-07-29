
import copy
import math

import numpy as np
import torch
import torch.nn as nn
from models.common.abstract_recommender import GeneralRecommender
import scipy.sparse as sp


class LightGCN_Encoder(GeneralRecommender):
    def __init__(self, config, dataset):

        super(LightGCN_Encoder, self).__init__(config, dataset)
        # load dataset info

        self.n_piRNAs = 10149
        self.n_diseases = 19
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)
        self.piRNA_count = self.n_piRNAs
        self.disease_count = self.n_diseases
        self.latent_size = config['embedding_size']

        self.n_layers = 3 if config['n_layers'] is None else config['n_layers']
        self.layers = [self.latent_size] * self.n_layers

        self.drop_ratio = 1.0
        self.drop_flag = True
        self.piRAN_e = np.load('../train data/piSim.npy')
        self.linear1 = nn.Linear(self.piRNA_count,self.latent_size)
        self.disease_e = np.load('../train data/DiseaSim.npy')
        self.linear2 = nn.Linear(self.disease_count, self.latent_size)
        self.sparse_norm_adj = self.get_norm_adj_mat().to(self.device)

    def _init_model(self):
        self.embedding_dict = nn.ParameterDict({
            'piRNA_emb': nn.Parameter(self.linear1(torch.FloatTensor(self.piRAN_e).to(self.device))),
            'disease_emb': nn.Parameter(self.linear2(torch.FloatTensor(self.disease_e).to(self.device)))
        })

        return self.embedding_dict

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of piRNAs and diseases.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_piRNAs + self.n_diseases,
                           self.n_piRNAs + self.n_diseases), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col+self.n_piRNAs),
                             [1]*inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row+self.n_piRNAs, inter_M_t.col),
                                  [1]*inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(self.device)
        return out * (1. / (1 - rate))

    def forward(self, inputs):
        initializer = nn.init.xavier_uniform_
        #with torch.no_grad():
        self.embedding_dict = nn.ParameterDict({
            'piRNA_emb': nn.Parameter(initializer(self.linear1(torch.FloatTensor(self.piRAN_e).to(self.device)))),
            'disease_emb': nn.Parameter(initializer(self.linear2(torch.FloatTensor(self.disease_e).to(self.device))))
        })
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    np.random.random() * self.drop_ratio,
                                    self.sparse_norm_adj._nnz()) if self.drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['piRNA_emb'], self.embedding_dict['disease_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        piRNA_all_embeddings = all_embeddings[:self.piRNA_count, :]
        disease_all_embeddings = all_embeddings[self.piRNA_count:, :]

        piRNAs, diseases = inputs[0], inputs[1]
        piRNA_embeddings = piRNA_all_embeddings[piRNAs, :]
        disease_embeddings = disease_all_embeddings[diseases, :]

        return piRNA_embeddings, disease_embeddings

    @torch.no_grad()
    def get_embedding(self):
        A_hat = self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['piRNA_emb'], self.embedding_dict['disease_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        piRNA_all_embeddings = all_embeddings[:self.piRNA_count, :]
        disease_all_embeddings = all_embeddings[self.piRNA_count:, :]

        return piRNA_all_embeddings, disease_all_embeddings
