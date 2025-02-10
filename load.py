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
import pandas as pd
import scipy.sparse as sp
import torch
import os

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def accuracy(output, labels):
    probs = torch.exp(output)
    preds = torch.argmax(probs, dim = 1)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def load_multi_data(folder, att_file, edge_list_name):
    att = pd.read_csv(folder + att_file)
    edge_list = []
    for name in edge_list_name:
        edge_list.append(pd.read_csv(folder + name))

    #get y and x
    labels = np.array(att["ever_pos"])
    # b,c,d are age groups
    features = sp.csr_matrix(att[["b","c","d", "Gender", "BLACK_AFRICAN_AMERICAN",
                                  "OTHER", "ASIAN", "pp_degree","pp_cls_coef",
                                  "pp_exposure_numerator","hh_degree"
                                  ,"education_center" , "assisted_living" ,
                                  "healthcare" ,   "charities_homeless" ,  "detention_center" ,
                                  "commercial_offices" ,  "government","hh_exposure_numerator" ,
                                  "hhnum_edu" ,"ppdeg_homeless","ppdeg_edu"]])    #features = normalize_features(features)

    #get adj mat
    adj_list = []
    for edge in edge_list:
        #get row col idx for adj matrix
        row_idx = []
        col_idx = []
        for i in range(edge.shape[0]):
            id_from = edge.iloc[i,0]
            id_to = edge.iloc[i,1]
            row_id = att.index[att["EventID"] == id_from]
            row_idx.append(row_id[0])
            col_id = att.index[att["EventID"] == id_to]
            col_idx.append(col_id[0])

        if edge.shape[1] == 2:
            adj = sp.coo_matrix((np.ones(edge.shape[0]), (row_idx, col_idx)), shape=(att.shape[0], att.shape[0]), dtype=np.float32)

        elif edge.shape[1] == 3:
            adj = sp.coo_matrix((np.array(edge.iloc[:,2]), (row_idx, col_idx)), shape=(att.shape[0], att.shape[0]), dtype=np.float32)

        #make adj symmetric
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        #normaliza adj
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        adj_list.append(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    dim = len(labels)
    idx_train = range(1585)
    idx_test = range(1585, dim)

    return adj_list, features, labels, idx_train, idx_test

def load_data(folder, att_file, edge_name):
    att = pd.read_csv(folder + att_file)
    edge_list = pd.read_csv(folder + edge_name)

    #get y and x
    labels = np.array(att["y"])
    # b,c,d are age groups
    features = sp.csr_matrix(att[["b","c","d", "Gender", "BLACK_AFRICAN_AMERICAN",
                                  "OTHER", "ASIAN", "pp_degree","pp_cls_coef",
                                  "pp_exposure_numerator","hh_degree"
                                  ,"education_center" , "assisted_living" ,
                                  "healthcare" ,   "charities_homeless" ,  "detention_center" ,
                                  "commercial_offices" ,  "government","hh_exposure_numerator" ,
                                  "hhnum_edu" ,"ppdeg_homeless","ppdeg_edu" ]])
    #features = normalize_features(features)

    #get adj mat
    row_idx = []
    col_idx = []
    for i in range(edge_list.shape[0]):
        id_from = edge_list.iloc[i,0]
        id_to = edge_list.iloc[i,1]
        row_id = att.index[att["EventID"] == id_from]
        row_idx.append(row_id[0])
        col_id = att.index[att["EventID"] == id_to]
        col_idx.append(col_id[0])

    if edge_list.shape[1] == 2:
        adj = sp.coo_matrix((np.ones(edge_list.shape[0]), (row_idx, col_idx)), shape=(att.shape[0], att.shape[0]), dtype=np.float32)


        #make adj symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        #normaliza adj
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    dim = len(labels)
    idx_train = range(1585)
    idx_test = range(1585, dim)

    return adj, features, labels, idx_train, idx_test