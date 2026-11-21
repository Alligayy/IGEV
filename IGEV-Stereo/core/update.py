import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

# 从最细尺度的 GRU 隐状态回归单通道视差残差，空间尺寸与当前视差一致。
class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

# 卷积 GRU 保留每个位置的隐状态。来自左图编码器的 cz、cr、cq 是可跨迭代复用的上下文项。
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):

        # 多个输入已对齐到同一分辨率，按通道与上一轮隐状态拼接。
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        # 重置门 r 对旧状态逐元素加权以生成候选状态 q，更新门 z 决定新状态中 q 的占比。
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)
        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, *x):
        # horizontal
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

# 将当前位置的视差与在两个相关体中查询到的匹配特征编码为 GRU 输入。
class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        # 每层两种体各取 2r+1 个样本，几何样本为 8 通道，全相关样本为 1 通道，默认共 162 个通道。
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        # 127 个融合通道再拼接原始视差，让 GRU 同时收到编码特征和当前估计值。
        return torch.cat([out, disp], dim=1)

def pool2x(x):
    # 将较细层的隐状态汇聚到相邻较粗层的网格。
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

# def interp(x, dest):
#     interp_args = {'mode': 'bilinear', 'align_corners': True}
#     return F.interpolate(x, dest.shape[2:], **interp_args)

def interp(x, dest):
    # 以 FP32 双线性插值将粗尺度隐状态对齐到目标层，再恢复输入的数据类型。
    original_dtype = x.dtype
    x_fp32 = x.float()
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    with torch.cuda.amp.autocast(enabled=False):
        output_fp32 = F.interpolate(x_fp32, dest.shape[2:], **interp_args)
    if original_dtype != torch.float32:
        output = output_fp32.to(original_dtype)
    else:
        output = output_fp32
    return output

class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        # hidden_dims 从粗到细排列。默认三层 GRU 的工作分辨率为 1/16、1/8、1/4。
        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr=None, disp=None, iter04=True, iter08=True, iter16=True, update=True):

        # net 和 inp 从细到粗排列。单轮按粗到细原地更新，较细层可以读取本轮刚更新的粗层。
        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            # 中间层接收上一轮细层的池化结果，启用第三层时还接收本轮粗层的插值结果。
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            # 最细层直接读取当前视差和局部匹配特征，并融合中间层提供的上下文信息。
            motion_features = self.encoder(disp, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        # 最细层分别输出视差改变量和用于预测上采样权重的 32 通道特征。
        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp
