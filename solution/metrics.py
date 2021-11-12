import torch


def mrr(F, G, offset=0):

    # distance between all pairs
    D = torch.cdist(G, F)
    # ranking of the distances
    rank = torch.argsort(torch.argsort(D)).diagonal(offset) + 1
    
    return torch.mean(1.0 / (rank*1.0)).item()


def mapk(V, F, G, k=10):

    assert V.shape[0] == F.shape[0] == G.shape[0] >= k
    assert F.shape[1] == G.shape[1]

    V_norm = torch.nn.functional.normalize(V)

    S = torch.matmul(V_norm, V_norm.t())
    R = (S > 0.3).float()
    D = torch.cdist(G, F)

    n,_ = D.shape
    indices = torch.argsort(D)[:,:k]
    offset = indices.new(range(0, n*n, n))
    R_top_k = torch.take(R, indices + offset[:, None])
    
    precision = torch.cumsum(R_top_k, dim=-1) / R_top_k.new(range(1, 1+k))
    average_precision = torch.sum(precision * R_top_k, dim=-1) / (torch.sum(R, dim=-1) + R.new([1e-12]))
    return torch.mean(average_precision).item()
