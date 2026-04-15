import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing, cross_modal_feature_fusion

# 添加调试信息
print("=== 正在使用MICA 模型（跨模态特征融合版本） ===")


def intra_modal_contrastive_loss(view1, view2, view3, temperature=1.0):  # 固定温度参数为1
    """
    计算模态内对比损失 (InfoNCE Loss)
    公式: L = -log(exp(sim(z_i, z_j)/τ) / (∑_{k≠i} exp(sim(z_i, z_k)/τ))

    参数:
        view1, view2, view3: 同一模态的三个视图表示 [batch_size, feature_dim]
        temperature: 温度参数τ，固定为1.0

    返回:
        三个视图两两之间的对比损失平均值
    """
    # 归一化特征向量
    view1 = F.normalize(view1, dim=1)
    view2 = F.normalize(view2, dim=1)
    view3 = F.normalize(view3, dim=1)

    # 计算三个视图两两之间的对比损失
    loss_12 = pairwise_contrastive_loss(view1, view2, temperature)
    loss_13 = pairwise_contrastive_loss(view1, view3, temperature)
    loss_23 = pairwise_contrastive_loss(view2, view3, temperature)

    # 返回平均损失
    return (loss_12 + loss_13 + loss_23) / 3


def pairwise_contrastive_loss(z1, z2, temperature=1.0):  # 固定温度参数为1
    """
    计算两个视图之间的对比损失
    """
    batch_size = z1.size(0)

    # 计算相似度矩阵
    sim_matrix = torch.mm(z1, z2.t()) / temperature

    # 正样本对是相同索引的位置
    labels = torch.arange(batch_size).to(z1.device)

    # 计算交叉熵损失
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


class Train_MICA:
    def __init__(self,
                 data,
                 datatype='SPOTS',
                 device=torch.device('cpu'),
                 random_seed=2022,
                 learning_rate=0.0001,
                 weight_decay=0.00,
                 epochs=1600,
                 dim_input=3000,
                 dim_output=64,
                 weight_factors=None,
                 fusion_beta=0.5  # 新增：融合系数参数
                 ):
        '''MICA Trainer with cross-modal feature fusion'''

        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.fusion_beta = fusion_beta  # 新增：融合系数

        # 固定温度参数为1
        self.temperature = 1.0

        # 处理weight_factors，固定对比学习权重为10
        if weight_factors is None:
            if self.datatype == 'SPOTS':
                self.epochs = 600
                self.weight_factors = [1, 5, 1, 1, 1, 1]  # 固定对比权重为10
            elif self.datatype == 'Stereo-CITE-seq':
                self.epochs = 1500
                self.weight_factors = [1, 10, 1, 10, 1, 1]  # 固定对比权重为10
            elif self.datatype == '10x':
                self.epochs = 1600
                self.weight_factors = [1, 5, 1, 10, 1, 1]  # 固定对比权重为10
            elif self.datatype == 'Spatial-epigenome-transcriptome':
                self.epochs = 1600
                self.weight_factors = [1, 5, 1, 1, 10, 10]  # 固定对比权重为10
        else:
            self.weight_factors = weight_factors

        # 处理邻接矩阵
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)

        # 获取三种邻接矩阵（转移到device）
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        self.adj_augmented_omics1 = self.adj['adj_augmented_omics1'].to(self.device)
        self.adj_augmented_omics2 = self.adj['adj_augmented_omics2'].to(self.device)

        # 处理特征矩阵（转移到device）
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        # 新增：跨模态特征融合
        print("=== 执行跨模态特征融合 ===")
        self.features_omics1_enhanced = cross_modal_feature_fusion(
            self.features_omics1, self.features_omics2, 
            self.adj_augmented_omics1, self.fusion_beta
        ).to(self.device)
        
        self.features_omics2_enhanced = cross_modal_feature_fusion(
            self.features_omics2, self.features_omics1,
            self.adj_augmented_omics2, self.fusion_beta
        ).to(self.device)

        print(f"原始RNA特征维度: {self.features_omics1.shape}")
        print(f"融合后RNA特征维度: {self.features_omics1_enhanced.shape}")
        print(f"原始ADT特征维度: {self.features_omics2.shape}")
        print(f"融合后ADT特征维度: {self.features_omics2_enhanced.shape}")

        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs

        # 输入特征维度
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

    def train(self):
        # 初始化模型
        self.model = Encoder_overall(
            self.dim_input1, self.dim_output1,
            self.dim_input2, self.dim_output2
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            # 前向传播（传入所有邻接矩阵和融合特征）
            results = self.model(
                self.features_omics1, self.features_omics2,
                self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_augmented_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2, self.adj_augmented_omics2,
                self.features_omics1_enhanced, self.features_omics2_enhanced  # 新增：融合特征
            )

            # 重构损失
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])

            # 一致性损失
            self.loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
            self.loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])

            # 模态内对比损失
            self.loss_contrast_omics1 = intra_modal_contrastive_loss(
                results['emb_latent_spatial_omics1'],
                results['emb_latent_feature_omics1'],
                results['emb_latent_augmented_omics1']
                # 不再传入温度参数，使用默认值1.0
            )

            self.loss_contrast_omics2 = intra_modal_contrastive_loss(
                results['emb_latent_spatial_omics2'],
                results['emb_latent_feature_omics2'],
                results['emb_latent_augmented_omics2']
                # 不再传入温度参数，使用默认值1.0
            )

            # 总损失 = 重构损失 + 一致性损失 + 对比损失
            loss = (
                    self.weight_factors[0] * self.loss_recon_omics1 +
                    self.weight_factors[1] * self.loss_recon_omics2 +
                    self.weight_factors[2] * self.loss_corr_omics1 +
                    self.weight_factors[3] * self.loss_corr_omics2 +
                    self.weight_factors[4] * self.loss_contrast_omics1 +
                    self.weight_factors[5] * self.loss_contrast_omics2
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Model training finished!\n")
        # 推理模式获取最终嵌入
        with torch.no_grad():
            self.model.eval()
            results = self.model(
                self.features_omics1, self.features_omics2,
                self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_augmented_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2, self.adj_augmented_omics2,
                self.features_omics1_enhanced, self.features_omics2_enhanced  # 新增：融合特征
            )

        # 归一化
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        # 输出结果
        output = {
            'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
            'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
            'MICA': emb_combined.detach().cpu().numpy(),
            'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
            'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
            'alpha': results['alpha'].detach().cpu().numpy(),
            # 新增：返回融合特征用于分析
            'features_omics1_enhanced': self.features_omics1_enhanced.detach().cpu().numpy(),
            'features_omics2_enhanced': self.features_omics2_enhanced.detach().cpu().numpy()
        }

        return output
    
    
    
        
    
    
      

    
        
    
    
