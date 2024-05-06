
import torch
import torch as t
from torch import nn
import torch.nn.functional as F
import torch_sparse
import numpy as np
import scipy.sparse as sp

from recbole.utils import InputType
from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class DCCF(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DCCF, self).__init__(config, dataset)
        
        self.user_num = self.n_users
        self.item_num = self.n_items
        self.embedding_size = config['embedding_size']

        rows = dataset.inter_feat[dataset.uid_field]
        cols = dataset.inter_feat[dataset.iid_field]

        new_rows = np.concatenate([rows, cols + self.user_num], axis=0)
        new_cols = np.concatenate([cols + self.user_num, rows], axis=0)
        plain_adj = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.user_num + self.item_num, self.user_num + self.item_num]).tocsr().tocoo()
        self.all_h_list = list(plain_adj.row)
        self.all_t_list = list(plain_adj.col)
        self.A_in_shape = plain_adj.shape
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).to(self.device)
        self.D_indices = torch.tensor([list(range(self.user_num + self.item_num)), list(range(self.user_num + self.item_num))], dtype=torch.long).to(self.device)
        self.all_h_list = torch.LongTensor(self.all_h_list).to(self.device)
        self.all_t_list = torch.LongTensor(self.all_t_list).to(self.device)
        self.G_indices, self.G_values = self._cal_sparse_adj()
        self.adaptive_masker = AdaptiveMask(head_list=self.all_h_list, tail_list=self.all_t_list, matrix_shape=self.A_in_shape, device=self.device)

        # hyper parameters
        self.layer_num = config['layer_num']
        self.intent_num = config['intent_num']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.temperature = config['temperature']


        # model parameters
        self.user_embeds = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_size)
        self.user_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.item_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)

        # train/test
        self.is_training = True
        self.final_embeds = None

        self._init_weight()

    def _init_weight(self):
        init(self.user_embeds.weight)
        init(self.item_embeds.weight)

    def _cal_sparse_adj(self):
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).to(self.device)
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).to(self.device)
        D_values = A_tensor.sum(dim=1).pow(-0.5)
        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        return G_indices, G_values

    def forward(self):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None, None, None, None

        all_embeds = [torch.concat([self.user_embeds.weight, self.item_embeds.weight], dim=0)]
        gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = [], [], [], []

        for i in range(0, self.layer_num):
            # Graph-based Message Passing
            gnn_layer_embeds = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])

            # Intent-aware Information Aggregation
            u_embeds, i_embeds = torch.split(all_embeds[i], [self.user_num, self.item_num], 0)
            u_int_embeds = torch.softmax(u_embeds @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeds = torch.softmax(i_embeds @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeds = torch.concat([u_int_embeds, i_int_embeds], dim=0)

            # Adaptive Augmentation
            gnn_head_embeds = torch.index_select(gnn_layer_embeds, 0, self.all_h_list)
            gnn_tail_embeds = torch.index_select(gnn_layer_embeds, 0, self.all_t_list)
            int_head_embeds = torch.index_select(int_layer_embeds, 0, self.all_h_list)
            int_tail_embeds = torch.index_select(int_layer_embeds, 0, self.all_t_list)
            G_graph_indices, G_graph_values = self.adaptive_masker(gnn_head_embeds, gnn_tail_embeds)
            G_inten_indices, G_inten_values = self.adaptive_masker(int_head_embeds, int_tail_embeds)
            gaa_layer_embeds = torch_sparse.spmm(G_graph_indices, G_graph_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])
            iaa_layer_embeds = torch_sparse.spmm(G_inten_indices, G_inten_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])

            # Aggregation
            gnn_embeds.append(gnn_layer_embeds)
            int_embeds.append(int_layer_embeds)
            gaa_embeds.append(gaa_layer_embeds)
            iaa_embeds.append(iaa_layer_embeds)
            all_embeds.append(gnn_layer_embeds + int_layer_embeds + gaa_layer_embeds + iaa_layer_embeds + all_embeds[i])

        all_embeds = torch.stack(all_embeds, dim=1)
        all_embeds = torch.sum(all_embeds, dim=1, keepdim=False)
        user_embeds, item_embeds = torch.split(all_embeds, [self.user_num, self.item_num], 0)
        self.final_embeds = all_embeds
        return user_embeds, item_embeds, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds
    
    def _cal_cl_loss(self, users, positems, negitems, gnn_emb, int_emb, gaa_emb, iaa_emb):
        users = torch.unique(users)
        items = torch.unique(torch.concat([positems, negitems]))
        cl_loss = 0.0
        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.user_num, self.item_num], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.user_num, self.item_num], 0)
            u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.user_num, self.item_num], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.user_num, self.item_num], 0)

            u_gnn_embs = u_gnn_embs[users]
            u_int_embs = u_int_embs[users]
            u_gaa_embs = u_gaa_embs[users]
            u_iaa_embs = u_iaa_embs[users]

            i_gnn_embs = i_gnn_embs[items]
            i_int_embs = i_int_embs[items]
            i_gaa_embs = i_gaa_embs[items]
            i_iaa_embs = i_iaa_embs[items]

            cl_loss += cal_infonce_loss(u_gnn_embs, u_int_embs, u_int_embs, self.temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(u_gnn_embs, u_gaa_embs, u_gaa_embs, self.temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(u_gnn_embs, u_iaa_embs, u_iaa_embs, self.temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_int_embs, i_int_embs, self.temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_gaa_embs, i_gaa_embs, self.temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_iaa_embs, i_iaa_embs, self.temperature) / u_gnn_embs.shape[0]
        
        return cl_loss
    
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        self.is_training = True
        # if self.restore_user_e is not None or self.restore_item_e is not None:
        #     self.restore_user_e, self.restore_item_e = None, None

        user_embeds, item_embeds, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = self.forward()
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        ancs, poss, negs = user, pos_item, neg_item

        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        cl_loss = self.cl_weight * self._cal_cl_loss(ancs, poss, negs, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds)

        return bpr_loss, reg_loss, cl_loss
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        self.is_training = False

        user_all_embeddings, item_all_embeddings, _, _, _, _ = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        self.is_training = False
        user = interaction[self.USER_ID]
        user_embeds, item_embeds, _, _, _, _ = self.forward()
        # get user embedding from storage variable
        u_embeddings = user_embeds[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, item_embeds.transpose(0, 1))

        return scores.view(-1)


class AdaptiveMask(nn.Module):
    """ Adaptively masking edges with learned weight (used in DCCF)
    """
    def __init__(self, head_list, tail_list, matrix_shape, device):
        """
        :param head_list: list of id about head nodes
        :param tail_list: list of id about tail nodes
        :param matrix_shape: shape of the matrix
        """
        super(AdaptiveMask, self).__init__()
        self.head_list = head_list
        self.tail_list = tail_list
        self.matrix_shape = matrix_shape
        self.device = device

    def forward(self, head_embeds, tail_embeds):
        """
        :param head_embeds: embeddings of head nodes
        :param tail_embeds: embeddings of tail nodes
        :return: indices and values (representing a augmented graph in torch_sparse fashion)
        """
        import torch_sparse
        head_embeddings = torch.nn.functional.normalize(head_embeds)
        tail_embeddings = torch.nn.functional.normalize(tail_embeds)
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = torch_sparse.SparseTensor(row=self.head_list, col=self.tail_list, value=edge_alpha, sparse_sizes=self.matrix_shape).to(self.device)
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        G_indices = torch.stack([self.head_list, self.tail_list], dim=0)
        G_values = D_scores_inv[self.head_list] * edge_alpha
        return G_indices, G_values


def reg_params(model):
	reg_loss = 0
	for W in model.parameters():
		reg_loss += W.norm(2).square()
	return reg_loss


def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
	pos_preds = (anc_embeds * pos_embeds).sum(-1)
	neg_preds = (anc_embeds * neg_embeds).sum(-1)
	return torch.sum(F.softplus(neg_preds - pos_preds))


def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
	""" InfoNCE Loss
	"""
	normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
	normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
	normed_all_embeds2 = all_embeds2 / t.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
	nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
	deno_term = t.log(t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
	cl_loss = (nume_term + deno_term).sum()
	return cl_loss