import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph


def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3):
    """
    Construct neighbor graphs, including feature graph, spatial graph and augmented feature graph.
    """

    # construct spatial neighbor graphs
    ################# spatial graph #################
    if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
        n_neighbors = 6
        # omics1
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1

    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2

    ################# feature graph #################
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2

    ################# augmented feature graph #################
    # 新增：构建增强特征图
    adj_augmented_omics1, adj_augmented_omics2 = construct_augmented_graph(
        adata_omics1, adata_omics2, feature_graph_omics1, feature_graph_omics2
    )
    adata_omics1.obsm['adj_augmented'] = adj_augmented_omics1
    adata_omics2.obsm['adj_augmented'] = adj_augmented_omics2

    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}

    return data


def construct_augmented_graph(adata_omics1, adata_omics2, feat_graph_omics1, feat_graph_omics2):
    """Constructing augmented feature graph based on expression profiles and spatial information"""

    # 使用空间邻接图确定邻居，但使用特征相似度计算权重
    spatial_graph_omics1 = adata_omics1.uns['adj_spatial']
    spatial_graph_omics2 = adata_omics2.uns['adj_spatial']

    features_omics1 = adata_omics1.obsm['feat']
    features_omics2 = adata_omics2.obsm['feat']

    # 转换空间图为稠密矩阵
    adj_spatial_omics1 = transform_adjacent_matrix(spatial_graph_omics1).toarray()
    adj_spatial_omics2 = transform_adjacent_matrix(spatial_graph_omics2).toarray()

    # 创建增强邻接矩阵
    adj_augmented_omics1 = np.zeros_like(adj_spatial_omics1)
    adj_augmented_omics2 = np.zeros_like(adj_spatial_omics2)

    # 为omics1构建增强图
    for i in range(adj_spatial_omics1.shape[0]):
        neighbors = np.where(adj_spatial_omics1[i] > 0)[0]
        if len(neighbors) > 0:
            # 计算余弦相似度
            cos_sim = sklearn.metrics.pairwise.cosine_similarity(
                features_omics1[i].reshape(1, -1),
                features_omics1[neighbors]
            )[0]
            # 应用指数变换增强差异
            S_ij = np.exp(2 - (1 - cos_sim))  # 因为cosine_dist = 1 - cos_sim
            # 归一化
            adj_augmented_omics1[i, neighbors] = S_ij / np.sum(S_ij)

    # 为omics2构建增强图
    for i in range(adj_spatial_omics2.shape[0]):
        neighbors = np.where(adj_spatial_omics2[i] > 0)[0]
        if len(neighbors) > 0:
            cos_sim = sklearn.metrics.pairwise.cosine_similarity(
                features_omics2[i].reshape(1, -1),
                features_omics2[neighbors]
            )[0]
            S_ij = np.exp(2 - (1 - cos_sim))
            adj_augmented_omics2[i, neighbors] = S_ij / np.sum(S_ij)

    return adj_augmented_omics1, adj_augmented_omics2


def cross_modal_feature_fusion(features_omics1, features_omics2, adj_augmented, beta=0.5):
    """
    跨模态特征融合
    features_omics1: 模态1的原始特征 [n_cells, n_features1]
    features_omics2: 模态2的原始特征 [n_cells, n_features2] 
    adj_augmented: 增强的邻接矩阵 [n_cells, n_cells]
    beta: 融合系数
    """
    # 确保邻接矩阵是稠密格式
    if isinstance(adj_augmented, torch.Tensor) and adj_augmented.is_sparse:
        adj_dense = adj_augmented.to_dense()
    else:
        adj_dense = adj_augmented
    
    # 特征融合: X_enhanced = X + β * (X_other @ A_w)
    features_enhanced = features_omics1 + beta * torch.matmul(adj_dense, features_omics2)
    
    return features_enhanced


def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode="connectivity", metric="correlation",
                               include_self=False):
    """Constructing feature neighbor graph according to expresss profiles"""

    feature_graph_omics1 = kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric,
                                            include_self=include_self)
    feature_graph_omics2 = kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric,
                                            include_self=include_self)

    return feature_graph_omics1, feature_graph_omics2


def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    """Constructing spatial neighbor graph according to spatial coordinates."""

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj


def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """Converting dense adjacent matrix to sparse adjacent matrix"""

    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)

    adj_spatial_omics1 = adj_spatial_omics1.toarray()  # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()

    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1 > 1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2 > 1, 1, adj_spatial_omics2)

    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1)  # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)

    ######################################## construct feature graph ########################################
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())

    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1 > 1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2 > 1, 1, adj_feature_omics2)

    # convert dense matrix to sparse matrix
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1)  # sparse adjacent matrix corresponding to feature graph
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2)

    ######################################## construct augmented feature graph ########################################
    # 新增：处理增强特征图
    adj_augmented_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_augmented'])
    adj_augmented_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_augmented'])

    # 增强图已经是归一化的权重矩阵，不需要对称化
    adj_augmented_omics1 = preprocess_graph(adj_augmented_omics1.numpy())
    adj_augmented_omics2 = preprocess_graph(adj_augmented_omics2.numpy())

    adj = {
        'adj_spatial_omics1': adj_spatial_omics1,
        'adj_spatial_omics2': adj_spatial_omics2,
        'adj_feature_omics1': adj_feature_omics1,
        'adj_feature_omics2': adj_feature_omics2,
        'adj_augmented_omics1': adj_augmented_omics1,  # 新增
        'adj_augmented_omics2': adj_augmented_omics2  # 新增
    }

    return adj


def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""LSI analysis (following the Seurat v3 approach)"""
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def tfidf(X):
    r"""TF-IDF normalization (following the Seurat v3 approach)"""
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'