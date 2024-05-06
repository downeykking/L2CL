import numpy as np
import torch
import torch.nn.functional as F

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv


class L2CL(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(L2CL, self).__init__(config, dataset)

        # load parameters info
        self.latent_dim = config['embedding_size']  # the embedding size of lightGCN
        self.n_layers = config['n_layers']  # the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # the weight decay for l2 normalization
        self.require_pow = config['require_pow']  # whether to require pow when regularization

        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']

        self.alpha = config['alpha']

        self.batch_size = config['batch_size']
        self.method = config['method']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def calculate_cl_loss_layer(self, layer_embedding, center_embedding, user, item, alpha, cl_mode="hetero"):
        layer_user_embeddings_all, layer_item_embeddings_all = torch.split(layer_embedding, [self.n_users, self.n_items])
        center_user_embeddings_all, center_item_embeddings_all = torch.split(center_embedding, [self.n_users, self.n_items])

        if cl_mode == "hetero":
            # user
            center_user_embeddings = center_user_embeddings_all[user]
            layer_user_embeddings = layer_item_embeddings_all[item]
            # item
            center_item_embeddings = center_item_embeddings_all[item]
            layer_item_embeddings = layer_user_embeddings_all[user]
        elif cl_mode == "homo":
            # user
            center_user_embeddings = center_user_embeddings_all[user]
            layer_user_embeddings = layer_user_embeddings_all[user]
            # item
            center_item_embeddings = center_item_embeddings_all[item]
            layer_item_embeddings = layer_item_embeddings_all[item]

        if self.batch_size == "batch":
            center_user_embeddings_all, center_item_embeddings_all = center_user_embeddings_all[user], center_item_embeddings_all[item]

        norm_user_emb1 = F.normalize(layer_user_embeddings)
        norm_user_emb2 = F.normalize(center_user_embeddings)
        norm_all_user_emb = F.normalize(center_user_embeddings_all)

        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        norm_item_emb1 = F.normalize(layer_item_embeddings)
        norm_item_emb2 = F.normalize(center_item_embeddings)
        norm_all_item_emb = F.normalize(center_item_embeddings_all)

        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = (alpha * ssl_loss_user + (1 - alpha) * ssl_loss_item)
        return ssl_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        center_embedding = embeddings_list[0]
        if self.n_layers >= 1:
            layer_embedding_1 = embeddings_list[1]

        if self.n_layers >= 2:
            layer_embedding_2 = embeddings_list[2]

        if self.n_layers >= 1:
            layer_embedding_all = embeddings_list[1:]
            layer_embedding_all = torch.stack(layer_embedding_all, dim=1)
            layer_embedding_all = torch.mean(layer_embedding_all, dim=1)

        if self.method == "u0-i0":
            assert self.n_layers == 0
            cl_loss = self.ssl_reg * self.calculate_cl_loss_layer(center_embedding, center_embedding, user, pos_item, self.alpha, cl_mode="hetero")
        elif self.method == "u1-i1":
            assert self.n_layers == 1
            cl_loss = self.ssl_reg * self.calculate_cl_loss_layer(layer_embedding_1, layer_embedding_1, user, pos_item, self.alpha, cl_mode="hetero")
        elif self.method == "u0-u2":
            assert self.n_layers >= 2
            cl_loss = self.ssl_reg * self.calculate_cl_loss_layer(layer_embedding_2, center_embedding, user, pos_item, self.alpha, cl_mode="homo")
        elif self.method == "u0-uall":
            assert self.n_layers >= 2
            cl_loss = self.ssl_reg * self.calculate_cl_loss_layer(layer_embedding_all, center_embedding, user, pos_item, self.alpha, cl_mode="homo")
        else:
            assert self.n_layers == 1
            cl_loss = self.ssl_reg * self.calculate_cl_loss_layer(layer_embedding_1, center_embedding, user, pos_item, self.alpha, cl_mode="hetero")

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)

        return mf_loss, self.reg_weight * reg_loss, cl_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, _ = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
