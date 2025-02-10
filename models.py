# This file is part of MARTRIX.
#
# MARTRIX is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MARTRIX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MARTRIX. If not, see <https://www.gnu.org/licenses/>.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = nn.Linear(nhid * nheads, nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x))
        return F.log_softmax(x, dim=1)



class FusionGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, adj_list, nheads):
        super(FusionGAT, self).__init__()

        self.dropout = dropout
        self.nheads = nheads
        self.adj_list = adj_list

        # Define list of GAT layers for each adjacency matrix
        self.attentions = nn.ModuleList()
        for i in range(len(adj_list)):
            att_list = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
            self.attentions.append(att_list)
            for k, attention in enumerate(self.attentions):
                self.add_module('adj{}, attention_{}'.format(i, k), attention)

        # Define linear layer for integration with L1 regularization
        self.integration_att = nn.Linear(nhid * nheads, nclass)
        self.fusion = nn.Linear(nclass * len(adj_list), nclass)
        self.l1_reg = nn.L1Loss(reduction='mean')

    def forward(self, x, adj_list):
        x = F.dropout(x, self.dropout, training=self.training)
        # Compute output for each adjacency matrix using GAT layers
        output_list = []
        for i, adj in enumerate(adj_list):
            x_i = torch.cat([att(x, adj) for att in self.attentions[i]], dim=1)
            x_i = F.dropout(x_i, self.dropout, training=self.training)
            x_i = F.elu(self.integration_att(x_i))
            output_list.append(x_i)
        output = torch.cat(output_list, dim=1)

        # Apply linear layer for integration with L1 regularization
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.fusion(output.view(output.size(0), -1))
        l1_loss = self.l1_reg(self.fusion.weight, torch.zeros_like(self.fusion.weight))
        return F.log_softmax(output, dim=1), l1_loss



class FusionGAT2(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout, alpha, adj_list, nheads):
        super(FusionGAT2, self).__init__()

        self.dropout = dropout
        self.nheads = nheads
        self.adj_list = adj_list

        # Define list of GAT layers for each adjacency matrix in attention 1
        self.attentions1 = nn.ModuleList()
        for i in range(len(adj_list)):
            att_list = nn.ModuleList([GraphAttentionLayer(nfeat, nhid1, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
            self.attentions1.append(att_list)
            for k, attention in enumerate(self.attentions1):
                self.add_module('adj{}, attention_layer1_{}'.format(i, k), attention)

        # Define linear layer for integration of multihead attention1
        self.integration_att1 = nn.Linear(nhid1 * nheads, nhid1)

        self.attentions2 = nn.ModuleList()
        for i in range(len(adj_list)):
            att_list2 = nn.ModuleList([GraphAttentionLayer(nhid1, nhid2, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
            self.attentions2.append(att_list2)
            for k, attention in enumerate(self.attentions2):
                self.add_module('adj{}, attention_layer2_{}'.format(i, k), attention)

        # Define linear layer for integration of multihead attention2
        self.integration_att2 = nn.Linear(nhid2 * nheads, nclass)

        #fusion layer with l1 penalty
        self.fusion_att = nn.Linear(nclass * len(adj_list), nclass)
        self.l1_reg = nn.L1Loss(reduction='mean')

    def forward(self, x, adj_list):
        x = F.dropout(x, self.dropout, training=self.training)
        # Compute output for each adjacency matrix using GAT layers
        output_list = []
        for i, adj in enumerate(adj_list):
            #attention layer 1
            x_i = torch.cat([att(x, adj) for att in self.attentions1[i]], dim=1)
            x_i = F.dropout(x_i, self.dropout, training=self.training)
            x_i = F.elu(self.integration_att1(x_i))

            #attention layer 2
            x_i = torch.cat([att(x_i, adj) for att in self.attentions2[i]], dim=1)
            x_i = F.dropout(x_i, self.dropout, training=self.training)
            x_i = F.elu(self.integration_att2(x_i))
            output_list.append(x_i)

        output = torch.cat(output_list, dim=1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.fusion_att(output.view(output.size(0), -1))
        l1_loss = self.l1_reg(self.fusion_att.weight, torch.zeros_like(self.fusion_att.weight))
        return F.log_softmax(output, dim=1), l1_loss



class FusionGAT3(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, fusion1_dim, nclass, dropout, alpha, adj_list, nheads):
        super(FusionGAT3, self).__init__()

        self.dropout = dropout
        self.nheads = nheads
        self.adj_list = adj_list

        # Define list of GAT layers for each adjacency matrix
        #att 1
        self.attentions1 = nn.ModuleList()
        for i in range(len(adj_list)):
            att_list = nn.ModuleList([GraphAttentionLayer(nfeat, nhid1, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
            self.attentions1.append(att_list)
            for k, attention in enumerate(self.attentions1):
                self.add_module('adj{}, attention_layer1_{}'.format(i, k), attention)

        # fusion1
        self.integration_att1 = nn.Linear(nhid1 * nheads, fusion1_dim)
        self.fusion_att1 = nn.Linear(fusion1_dim * len(adj_list), fusion1_dim)
        self.l1_reg1 = nn.L1Loss(reduction='mean')

        #att 2
        self.attentions2 = nn.ModuleList()
        for i in range(len(adj_list)):
            att_list = nn.ModuleList([GraphAttentionLayer(fusion1_dim, nhid2, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
            self.attentions2.append(att_list)
            for k, attention in enumerate(self.attentions2):
                self.add_module('adj{}, attention_layer2_{}'.format(i, k), attention)

        #fusion2
        self.integration_att2 = nn.Linear(nhid2 * nheads, nclass)
        self.fusion_att2 = nn.Linear(nclass * len(adj_list), nclass)
        self.l1_reg2 = nn.L1Loss(reduction='mean')


    def forward(self, x, adj_list):
        x = F.dropout(x, self.dropout, training=self.training)

        # Compute output for each adjacency matrix using GAT layers
        output_list = []
        for i, adj in enumerate(adj_list):
            x_i = x
            x_i = torch.cat([att(x, adj) for att in self.attentions1[i]], dim=1)
            x_i = F.dropout(x_i, self.dropout, training=self.training)
            x_i = F.elu(self.integration_att1(x_i))
            output_list.append(x_i)
        output = torch.cat(output_list, dim=1)

        # Apply linear layer for integration with L1 regularization
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.fusion_att1(output.view(output.size(0), -1))
        l1_loss1 = self.l1_reg1(self.fusion_att1.weight, torch.zeros_like(self.fusion_att1.weight))


        output_list2 = []
        for i, adj in enumerate(adj_list):
            x_i = torch.cat([att(output, adj) for att in self.attentions2[i]], dim=1)
            x_i = F.dropout(x_i, self.dropout, training=self.training)
            x_i = F.elu(self.integration_att2(x_i))
            output_list2.append(x_i)
        output2 = torch.cat(output_list2, dim=1)

        # Apply linear layer for integration with L1 regularization
        output2 = F.dropout(output2, self.dropout, training=self.training)
        output2 = self.fusion_att2(output2.view(output.size(0), -1))
        l1_loss2 = self.l1_reg2(self.fusion_att2.weight, torch.zeros_like(self.fusion_att2.weight))

        return F.log_softmax(output2, dim=1), l1_loss1 + l1_loss2

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)