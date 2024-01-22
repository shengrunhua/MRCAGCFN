import torch

def dist_mask(data,device):
    data=torch.tensor(data)
    data=data.to(device)
    tile_data = torch.unsqueeze(data, dim=1)
    # shape=(4,1,441,200)
    next_data = torch.unsqueeze(data, dim=-2)
    # shape=(4,441,1,200)
    minus = tile_data - next_data
    # shape=(4,441,441,200) 广播 各像素光谱维度距离
    a = -torch.sum(minus**2, -1)
    # shape=(4,441,441) 光谱维度降没(求和后降维)，光谱维度距离和
    dist = torch.exp(a/data.shape[2])
    # shape=(4,441,441) dist=e的a/S(200)次方，e的-光谱维度距离平均值次方
    dist = dist/torch.sum(dist, 2, keepdims=True)
    # dist除以“自身第三个维度降为1的矩阵(shape=(4,441,1))”,shape=(4,441,441)  归一化
    dist=dist+torch.eye(data.shape[1]).to(device)

    return dist.cpu()
    