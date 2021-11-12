import torch


def assignment_loss(V, F, G, c = 0.5, a = 0.5):

    assert V.shape[0] == F.shape[0] == G.shape[0]
    assert F.shape[1] == G.shape[1]

    # normalize ground truths
    V_norm = torch.nn.functional.normalize(V)
    # cossine similarities of all pairs
    S = torch.matmul(V_norm, V_norm.t())
    # indicator for all pairs
    I = (S == 0).float()
    # distance between all pairs
    D = torch.cdist(G, F)

    t1 = (1-a) * torch.sum(S * (D ** 2))
    t2 = a * torch.sum(torch.clamp(I * (c - (D ** 2)), min=0.))
    loss = t1 + t2

    return loss, t1, t2


def modified_loss(V, F, G, c = 0.5, a = 0.5):

    assert V.shape[0] == F.shape[0] == G.shape[0]
    assert F.shape[1] == G.shape[1]

    # normalize ground truths
    V_norm = torch.nn.functional.normalize(V)
    # cossine similarities of all pairs
    S = torch.matmul(V_norm, V_norm.t())
    # indicator for all pairs
    R = (S == 1).float()
    # distance between all pairs
    D = torch.cdist(G, F)

    t1 = (1-a) * torch.sum(R * (D ** 2))
    t2 = a * torch.sum(torch.clamp((1-S) * c - D, min=0.) ** 2)
    loss = t1 + t2

    return loss, t1, t2
