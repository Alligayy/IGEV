import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import MultiBasicEncoder, Feature
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
import time


try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

# 3D 沙漏网络在视差和图像空间三个维度上聚合匹配信息。
# 输入是 [B, 8, D, H/4, W/4]，输出恢复到相同的尺寸，作为几何编码体。
class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        # 每级先用步长为 2 的 3D 卷积同时缩小 D、H、W，再在当前尺度提取特征。
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        # 3D 转置卷积逐级恢复视差轴和空间分辨率，供解码端与同尺度的编码特征融合。
        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        # 拼接使通道数翻倍，先用 1×1×1 卷积压缩，再用 3×3×3 卷积聚合邻域信息。
        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        # 用同分辨率的左图 2D 特征为代价体生成门控权重，按通道和像素位置调制整条视差轴。
        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        # features 按 1/4、1/8、1/16、1/32 排列，编码端逐级用对应的图像特征引导。
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        # 解码端通过跳跃连接取回同尺度的细节，再用左图特征调制融合结果。
        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv

class IGEVStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        # 左图上下文编码器初始化多尺度 GRU，更新块在每轮迭代中复用。
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        # 每个尺度用卷积一次生成三组门控输入，默认三个尺度均使用 128 个隐藏通道。
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feature = Feature()

        # 左右图共享浅层卷积。1/4 特征补充匹配描述子，左图 1/2 特征还用于引导上采样。
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        # 初始视差的上采样权重由左图特征预测，先融合 1/4 和 1/2 特征，再生成全分辨率权重。
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        # 迭代视差的上采样权重由当前 GRU 特征和左图的 1/2 特征共同预测。
        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def freeze_bn(self):
        # 训练脚本调用这里，让二维 BatchNorm 使用已有的运行均值和方差。
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            # 每个全分辨率像素预测 9 个权重，对应低分辨率视差的 3×3 邻域。
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            # 将 1/4 网格上的视差乘以 4，换算为原图像素单位后再做加权上采样。
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        # 输入为已极线校正的左右图像 [B, 3, H, W]，将像素值从 [0, 255] 映射到 [-1, 1]。
        # 评估和演示脚本会先将 H、W 填充到 32 的倍数，使多尺度特征与 3D 沙漏的跳跃连接对齐。
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            features_left = self.feature(image1)
            features_right = self.feature(image2)
            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)
            # 在 1/4 尺度拼接骨干特征和浅层细节，通道数为 48 + 48 = 96。
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))
            # 每个视差候选 d 对齐左图 x 和右图 x-d。96 个描述子通道分为 8 组，逐组计算相关性。
            # D = max_disp//4 是 1/4 网格上的候选数，默认 192 对应 48 个候选。
            # D 取 8 的倍数可与 3D 沙漏的三次二倍下采样和上采样对齐。
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
            geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

            # 将几何编码体压成每个视差候选的分数，沿视差轴做 softmax。
            # 用视差概率的期望得到初始估计 [B, 1, H/4, W/4]，数值仍以 1/4 网格为单位。
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            init_disp = disparity_regression(prob, self.args.max_disp//4)
            
            del prob, gwc_volume

            if not test_mode:
                # 训练对初始视差单独监督，因此也为它生成一套全分辨率上采样权重。
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

            # 默认按 1/4、1/8、1/16 取左图上下文。每个尺度有两个输出，分别用于
            # 初始化隐状态，以及预计算更新门、重置门和候选状态的上下文项 (cz, cr, cq)。
            # 这些上下文项由左图确定，在本次前向传播的所有迭代中复用。
            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]


        # 一次性构建几何编码体和同一行全相关体的金字塔，迭代时只查询当前估计附近的特征。
        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        # [B, H/4, W/4, 1] 中每个像素存的是左图横坐标，用于计算右图匹配位置 x-d。
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = init_disp
        disp_preds = []

        # 根据当前视差查询局部匹配证据，更新多尺度 GRU，再给当前估计加上视差残差。
        for itr in range(iters):
            # 截断经过上一轮视差估计的梯度路径，GRU 隐状态仍在迭代间传递并保留梯度连接。
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)

            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            # 训练保留每轮的全分辨率预测，推理只在最后一轮执行上采样。
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        # 初始预测也恢复到 [B, 1, H, W]，与各轮预测使用同样的原图像素单位。
        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        return init_disp, disp_preds
