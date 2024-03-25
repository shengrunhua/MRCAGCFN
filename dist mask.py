import torch

def dist_mask(data, device):
    data = torch.tensor(data)
    data = data.to(device)
    tile_data = torch.unsqueeze(data, dim=1)
    next_data = torch.unsqueeze(data, dim=-2)
    minus = tile_data - next_data
    a = -torch.sum(minus ** 2, -1)
    dist = torch.exp(a / data.shape[2])
    dist = dist / torch.sum(dist, 2, keepdims = True)
    dist = dist + torch.eye(data.shape[1]).to(device)

    return dist.cpu()  
