import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from .preprocess import pca
import matplotlib.pyplot as plt


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """Clustering using the mclust algorithm"""
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, key='emb', add_key='MICA', method='mclust', start=0.1, end=3.0, increment=0.01,
               use_pca=False, n_comps=20):
    """Spatial clustering based the latent representation"""
    if use_pca:
        adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)

    if method == 'mclust':
        if use_pca:
            adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
        else:
            adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
        adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
        if use_pca:
            res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end,
                             increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
        if use_pca:
            res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end,
                             increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['louvain']


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''Searching corresponding resolution according to given cluster number'''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res


def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
    """Plotting weight values"""
    import pandas as pd

    # 修改为处理三个视图的权重
    df = pd.DataFrame(columns=['View1', 'View2', 'View3', 'label'])
    df['View1'], df['View2'], df['View3'] = alpha[:, 0], alpha[:, 1], alpha[:, 2]
    df['label'] = label

    # 转换为长格式
    df_long = df.melt(id_vars=['label'], var_name='View', value_name='Weight value')

    # 绘制
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(
        data=df_long,
        x='label',
        y='Weight value',
        hue="View",
        split=True,
        inner="quart",
        linewidth=1
    )
    ax.set_title(f'Attention Weights: {modality1}')
    plt.legend(title='View Type', loc='upper right')
    plt.tight_layout()
    plt.show()


def evaluate_fusion_effect(original_features, fused_features, labels, method='silhouette'):
    """
    评估特征融合效果
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
    if method == 'silhouette':
        original_score = silhouette_score(original_features, labels)
        fused_score = silhouette_score(fused_features, labels)
    elif method == 'calinski_harabasz':
        original_score = calinski_harabasz_score(original_features, labels)
        fused_score = calinski_harabasz_score(fused_features, labels)
    else:
        raise ValueError("Unsupported evaluation method")
    
    improvement = (fused_score - original_score) / original_score * 100
    
    print(f"Original {method} score: {original_score:.4f}")
    print(f"Fused {method} score: {fused_score:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    
    return original_score, fused_score, improvement


def visualize_fusion_comparison(original_emb, fused_emb, labels, title1="Original Features", title2="Fused Features"):
    """
    可视化原始特征和融合特征的对比
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始特征
    scatter1 = ax1.scatter(original_emb[:, 0], original_emb[:, 1], c=labels, cmap='tab10', s=10)
    ax1.set_title(title1)
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    plt.colorbar(scatter1, ax=ax1)
    
    # 融合特征
    scatter2 = ax2.scatter(fused_emb[:, 0], fused_emb[:, 1], c=labels, cmap='tab10', s=10)
    ax2.set_title(title2)
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.show()