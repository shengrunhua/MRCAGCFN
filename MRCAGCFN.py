import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class MRCAGCFN(nn.Module):
    def __init__(self, S, l, class_num, hidden, device):
        super(MRCAGCFN, self).__init__()
        self.S = S
        self.l = l
        self.device = device
        
        self.hidden = int(hidden)
        
        self.spectral = nn.Sequential(nn.Conv2d(S, hidden * 2, (1, 1)),
                                    nn.BatchNorm2d(hidden * 2),
                                    mish(),
                                    nn.Conv2d(hidden * 2, hidden, (1, 1)),
                                    nn.BatchNorm2d(hidden),
                                    mish())
        
        self.conv1 = nn.ModuleList([DeformConv2d(hidden, hidden, kernel_size = (3, 3), padding = 1) for i in range(3)])
        self.conv2 = nn.ModuleList([DeformConv2d(hidden, hidden, kernel_size = (5, 5), padding = 2) for i in range(3)])
        self.conv3 = nn.ModuleList([DeformConv2d(hidden, hidden, kernel_size = (7, 7), padding = 3) for i in range(3)])
        self.bnc = nn.ModuleList([nn.BatchNorm2d(hidden) for i in range(9)])
        
        self.gcn = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(3)])
        self.aff = nn.ModuleList([nn.Conv1d(l ** 2, l ** 2, kernel_size = 2, dilation = hidden, groups = l ** 2) for i in range(3)])
        self.bng = nn.ModuleList([nn.BatchNorm1d(l ** 2) for i in range(3)])
        
        self.fpc = nn.Conv1d(hidden, hidden, kernel_size = (l ** 2), padding = 0, groups = hidden)
        self.bnfpc = nn.BatchNorm1d(hidden)
        self.fpaff = nn.Conv1d(1, 1, kernel_size = 2, padding = 0, dilation = hidden)
        self.bnfpg = nn.BatchNorm1d(1)
        
        self.output = nn.Linear(hidden, class_num)
    
    
    def AFGCM(self, x_g, dist, i):
        x_g = self.gcn[i](x_g)
        res_x_g = x_g
        x_g = torch.bmm(dist, x_g)
        x_g = torch.cat((x_g, res_x_g), dim=2)
        x_g = self.aff[i](x_g)
        x_g = self.bng[i](x_g)
        x_g = mish()(x_g)
        
        return x_g
    
    def MRCM(self, x_c, i):
        x_c = torch.transpose(x_c, 1, 3)
        offset0 = torch.rand(x_c.size(0), 2 * 3 * 3, x_c.size(2), x_c.size(3)).to(self.device) * 2 - 1
        x_c0 = self.conv1[i](x_c, offset0)
        x_c0 = self.bnc[i * 3 + 0](x_c0)
        x_c0 = torch.transpose(x_c0, 1, 3)
        x_c0 = mish()(x_c0)
        offset1 = torch.rand(x_c.size(0), 2 * 5 * 5, x_c.size(2), x_c.size(3)).to(self.device) * 2 - 1
        x_c1 = self.conv2[i](x_c, offset1)
        x_c1 = self.bnc[i * 3 + 1](x_c1)
        x_c1 = torch.transpose(x_c1, 1, 3)
        x_c1 = mish()(x_c1)
        offset2 = torch.rand(x_c.size(0), 2 * 7 * 7, x_c.size(2), x_c.size(3)).to(self.device) * 2 - 1
        x_c2 = self.conv3[i](x_c, offset2)
        x_c2 = self.bnc[i * 3 + 2](x_c2)
        x_c2 = torch.transpose(x_c2, 1, 3)
        x_c2 = mish()(x_c2)
        x_c = (x_c0 + x_c1 + x_c2) / 3
        return x_c

    def fusion(self, x_g, x_c, i):
        x = (x_g + x_c) / 2
        
        return x
    
    def ALFPM(self, x, dist):
        x_g = x
        x_c = torch.transpose(x, 1, 2)
        x_c = self.fpc(x_c)
        x_c = self.bnfpc(x_c)
        x_c = torch.transpose(x_c, 1, 2)
        x_c = mish()(x_c)
        res_x_g = x_g[:, int((x_g.size(1) - 1) / 2), :].unsqueeze(1)
        dist = dist[:, int((x_g.size(1) - 1) / 2), :].unsqueeze(1)
        x_g = torch.bmm(dist, x_g)
        x_g = torch.cat((x_g, res_x_g), dim=2)
        x_g = self.fpaff(x_g)
        x_g = self.bnfpg(x_g)
        x_g = mish()(x_g)
        x = (x_c + x_g) / 2
        x = x.squeeze(1)
        x = self.output(x)
        return x
        
        
    def forward(self,x,dist):
        x = x.to(torch.float32)
        dist = dist.to(torch.float32)
        x = x.reshape(-1, self.l, self.l, x.shape[2])
        x = torch.transpose(x, 1, 3)
        x = self.spectral(x)
        x = torch.transpose(x, 1, 3)
        x = x.reshape(-1, self.l ** 2, x.shape[3])
        for i in range(3):
            x_g = x
            x_c = x.reshape(-1, self.l, self.l, x.shape[2])
            x_g = self.AFGCM(x_g, dist, i)
            x_c = self.MRCM(x_c, i)
            x_c = x_c.reshape(-1, self.l ** 2, x_c.shape[3])
            x = self.fusion(x_g, x_c, i)
        
        x = self.ALFPM(x, dist)
        
        return x
