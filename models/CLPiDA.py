import torch
import torch.nn as nn
import torch.nn.functional as F
from models.LightGCN import LightGCN_Encoder
from models.common.loss import L2Loss
from models.common.abstract import General

class CLPiDA(General):
    def __init__(self, config, dataset):
        super(CLPiDA, self).__init__(config, dataset)
        self.piRNA_count = self.n_piRNAs
        self.disease_count = self.n_diseases
        self.latent_size = config['embedding_size']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.linear_weight = config['linear_weight']

        self.online_encoder = LightGCN_Encoder(config, dataset)
        self.predictor = nn.Linear(self.latent_size, self.latent_size)
        self.reg_loss = L2Loss()

    def forward(self):
        p_online, d_online = self.online_encoder.get_embedding()
        with torch.no_grad():
            p_target, d_target = self.online_encoder.get_embedding()
            p_target.detach()
            d_target.detach()
            p_target = F.dropout(p_target, self.dropout)
            d_target = F.dropout(d_target, self.dropout)
        return p_online, p_target, d_online, d_target

    @torch.no_grad()
    def get_embedding(self):
        p_online, d_online = self.online_encoder.get_embedding()
        return self.predictor(p_online), p_online, self.predictor(d_online), d_online

    def loss_fn(self, p, z):  # negative cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def calculate_loss(self, interaction):
        p_all_online, p_all_target, d_all_online, d_all_target = self.forward()

        piRNAs, diseases = interaction[0], interaction[1]
        p_online = p_all_online[piRNAs, :]
        d_online = d_all_online[diseases, :]
        p_target = p_all_target[piRNAs, :]
        d_target = d_all_target[diseases, :]

        reg_loss = self.reg_loss(p_online, d_online)

        p_online, d_online = self.predictor(p_online), self.predictor(d_online)

        loss_pd = self.loss_fn(p_online, d_target)/2
        loss_dp = self.loss_fn(d_online, p_target)/2

        linear_loss = .0
        for param in self.predictor.parameters():
            linear_loss += torch.norm(param, 1) ** 2

        return loss_pd + loss_dp + self.reg_weight * reg_loss + self.linear_weight * linear_loss

    def full_sort_predict(self, interaction):
        piRNA = interaction[0]

        p_online, p_target, d_online, d_target = self.get_embedding()
        score_mat_pd = torch.matmul(p_online[piRNA], d_target.transpose(0, 1))
        score_mat_dp = torch.matmul(p_target[piRNA], d_online.transpose(0, 1))
        scores = score_mat_pd + score_mat_dp

        return scores

    def getembding(self):
        p_online, p_target, d_online, d_target = self.get_embedding()
        return p_online,p_target,d_online,d_target

    def depenttest(self,test_pd):
        n_piRNAs = len(test_pd)
        n_diseases = len(test_pd[0])
        self.piRNA_embedding = nn.Embedding(n_piRNAs, self.latent_size)
        self.disease_embedding = nn.Embedding(n_diseases, self.latent_size)
        p_online = F.dropout(self.piRNA_embedding.weight, 0.0)
        d_online = F.dropout(self.disease_embedding.weight, 0.0)
        p_target = self.predictor(p_online)
        d_target = self.predictor(d_online)
        p_online, p_target, d_online, d_target = self.get_embedding()
        score_mat_pd = torch.matmul(p_online, d_target.transpose(0, 1))
        score_mat_dp = torch.matmul(p_target, d_online.transpose(0, 1))
        scores = score_mat_pd + score_mat_dp

        return scores