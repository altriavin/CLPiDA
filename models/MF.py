import torch
import torch.nn as nn
import numpy as np

from models.common.abstract import General
from models.common.loss import BPRLoss, EmbLoss
import torch.nn.functional as F


class MF(General):
    def __init__(self, config, dataset):
        super(MF, self).__init__(config, dataset)

        self.embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)
        self.piRNA_count = self.n_piRNAs
        self.disease_count = self.n_diseases
        # define layers and loss
        self.piRNA_embedding = nn.Embedding(self.n_piRNAs, self.embedding_size)
        self.disease_embedding = nn.Embedding(self.n_diseases, self.embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()


    def get_piRNA_embedding(self, piRNA):
        return self.piRNA_embedding(piRNA)

    def get_disease_embedding(self, disease):
        return self.disease_embedding(disease)

    @torch.no_grad()
    def get_embedding(self):
        piRNA_num = torch.LongTensor(np.arange(10149))
        disease_num = torch.LongTensor(np.arange(19))
        return self.get_piRNA_embedding(piRNA_num),self.get_disease_embedding(disease_num)

    def forward(self, dropout=0.0):
        piRNA_e = F.dropout(self.piRNA_embedding.weight, dropout)
        disease_e = F.dropout(self.disease_embedding.weight, dropout)
        return piRNA_e, disease_e

    def calculate_loss(self, interaction):
        piRNA = interaction[0]
        pos_disease = interaction[1]
        neg_disease = interaction[2]

        piRNA_embeddings, disease_embeddings = self.forward()
        piRNA_e = piRNA_embeddings[piRNA, :]
        pos_e = disease_embeddings[pos_disease, :]
        neg_e = self.get_disease_embedding(neg_disease)
        pos_disease_score, neg_disease_score = torch.mul(piRNA_e, pos_e).sum(dim=1), torch.mul(piRNA_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_disease_score, neg_disease_score)
        reg_loss = self.reg_loss(piRNA_e, pos_e, neg_e)
        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def full_sort_predict(self, interaction):
        piRNA = interaction[0]
        piRNA_e = self.get_piRNA_embedding(piRNA)
        all_disease_e = self.disease_embedding.weight
        score = torch.matmul(piRNA_e, all_disease_e.transpose(0, 1))
        return score
