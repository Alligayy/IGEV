import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler


# 为每个左图像素缓存两条一维匹配序列。几何编码体按视差索引，
# 全相关体按右图横坐标索引。两者联合为视差迭代提供匹配特征。
class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        # 计算同一行内每个左图位置与所有右图位置的描述子点积。
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        # 把左图像素合并入批次维。每个像素对应 [C, 1, D] 的几何序列和 [1, 1, W2] 的相关序列。
        # 保留高度为 1 的维度，便于用 grid_sample 查询连续坐标。
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        # 平均池化仅压缩候选轴。金字塔第 i 层的一个采样步长对应原序列的 2**i 个位置。
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)




    def __call__(self, disp, coords):
        # disp 为 [B, 1, H/4, W/4] 的当前视差，coords 保存 1/4 网格上的左图横坐标。
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            # 以当前视差在该层的位置为中心，等距采样 2r+1 个视差候选。
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            # 全相关体用右图横坐标索引，采样中心为 (x_left-disp)/2**i，偏移 dx 已处于该层尺度。
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        # 各层的候选位置摊入通道维，返回 [B, num_levels*(2r+1)*(C+1), H/4, W/4]。
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    
    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        # 沿特征通道求和，保留共同的行索引和左右图各自的列索引，得到 [B, H, W1, W2]。
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr