import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Encoder_overall(Module):
    """Overall encoder for MICA with intra-modal contrastive learning"""

    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dropout=0.0,
                 act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act

        # 每个模态有三个编码器
        self.encoder_omics1_spatial = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.encoder_omics1_feature = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.encoder_omics1_augmented = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)

        self.encoder_omics2_spatial = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.encoder_omics2_feature = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.encoder_omics2_augmented = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)

        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)

        # 模态内注意力融合（三个表示）
        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1, num_views=3)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2, num_views=3)

        # 跨模态注意力保持不变
        self.atten_cross = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)

    def forward(self, features_omics1, features_omics2,
                adj_spatial_omics1, adj_feature_omics1, adj_augmented_omics1,
                adj_spatial_omics2, adj_feature_omics2, adj_augmented_omics2):
        # Omics1 的三个图表示
        emb_latent_spatial_omics1 = self.encoder_omics1_spatial(features_omics1, adj_spatial_omics1)
        emb_latent_feature_omics1 = self.encoder_omics1_feature(features_omics1, adj_feature_omics1)
        emb_latent_augmented_omics1 = self.encoder_omics1_augmented(features_omics1, adj_augmented_omics1)

        # Omics2 的三个图表示
        emb_latent_spatial_omics2 = self.encoder_omics2_spatial(features_omics2, adj_spatial_omics2)
        emb_latent_feature_omics2 = self.encoder_omics2_feature(features_omics2, adj_feature_omics2)
        emb_latent_augmented_omics2 = self.encoder_omics2_augmented(features_omics2, adj_augmented_omics2)

        # 模态内注意力融合（三个表示）
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(
            emb_latent_spatial_omics1,
            emb_latent_feature_omics1,
            emb_latent_augmented_omics1
        )

        emb_latent_omics2, alpha_omics2 = self.atten_omics2(
            emb_latent_spatial_omics2,
            emb_latent_feature_omics2,
            emb_latent_augmented_omics2
        )

        # 跨模态注意力融合
        emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_latent_omics1, emb_latent_omics2)

        # 重构
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        # 一致性编码
        emb_latent_omics1_across_recon = self.encoder_omics2_spatial(
            self.decoder_omics2(emb_latent_omics1, adj_spatial_omics2),
            adj_spatial_omics2
        )

        emb_latent_omics2_across_recon = self.encoder_omics1_spatial(
            self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1),
            adj_spatial_omics1
        )

        results = {
            'emb_latent_omics1': emb_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined,
            'emb_recon_omics1': emb_recon_omics1,
            'emb_recon_omics2': emb_recon_omics2,
            'emb_latent_omics1_across_recon': emb_latent_omics1_across_recon,
            'emb_latent_omics2_across_recon': emb_latent_omics2_across_recon,
            'alpha_omics1': alpha_omics1,
            'alpha_omics2': alpha_omics2,
            'alpha': alpha_omics_1_2,
            # 新增：返回三个视图的表示用于对比学习
            'emb_latent_spatial_omics1': emb_latent_spatial_omics1,
            'emb_latent_feature_omics1': emb_latent_feature_omics1,
            'emb_latent_augmented_omics1': emb_latent_augmented_omics1,
            'emb_latent_spatial_omics2': emb_latent_spatial_omics2,
            'emb_latent_feature_omics2': emb_latent_feature_omics2,
            'emb_latent_augmented_omics2': emb_latent_augmented_omics2
        }

        return results


class Encoder(Module):
    """Modality-specific GNN encoder"""

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x


class Decoder(Module):
    """Modality-specific GNN decoder"""

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x


class AttentionLayer(Module):
    """Attention layer with support for multiple views"""

    def __init__(self, in_feat, out_feat, num_views=2, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_views = num_views

        # 参数矩阵
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, *embeddings):
        # 将所有嵌入堆叠为 [num_views, num_cells, feat_dim]
        self.emb = torch.stack(embeddings, dim=1)

        # 计算注意力分数
        v = F.tanh(torch.matmul(self.emb, self.w_omega))
        vu = torch.matmul(v, self.u_omega)
        alpha = F.softmax(torch.squeeze(vu, dim=-1), dim=1)  # [num_cells, num_views]

        # 加权融合
        emb_combined = torch.sum(self.emb * alpha.unsqueeze(-1), dim=1)

        return emb_combined, alpha