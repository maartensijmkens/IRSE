import torch


def random_sampling(F, G):

    n = F.shape[0]

    # distances of projections of all pairs 
    D = torch.cdist(G, F)

    # distances of corresponding pairs
    D_xz = torch.diag(D)

    # select random negatives
    i = torch.unsqueeze((torch.randint(1, n, size=(n,)) + torch.arange(n)) % n, dim=0)
    j = torch.unsqueeze((torch.randint(1, n, size=(n,)) + torch.arange(n)) % n, dim=0)

    # get distances to random negatives
    D_xn = torch.gather(D, -1, i)
    D_nz = torch.gather(D, -2, j)
   
    return D_xz, D_xn, D_nz


def hard_sampling(V, F, G):

    assert V.shape[0] == F.shape[0] == G.shape[0]
    assert F.shape[1] == G.shape[1]    

    # normalize ground truths
    V_norm = torch.nn.functional.normalize(V)

    # cossine similarities of all pairs
    S = torch.matmul(V_norm, V_norm.t())

    # distances of projections of all pairs 
    D = torch.cdist(G, F)

    # distances of corresponding pairs
    D_xz = torch.diag(D)

    # set distances of corresponding pairs to max value
    D_masked = D + 2*torch.diag(torch.ones_like(D_xz))

    # get distances to hard negatives
    D_xn, i = torch.min(D_masked, dim=-1)
    D_nz, j = torch.min(D_masked, dim=-2)

    i = torch.unsqueeze(i, dim=0)
    j = torch.unsqueeze(j, dim=0)

    # get similarities with hard negatives
    S_xn = torch.gather(S, -1, i)
    S_nz = torch.gather(S, -2, j)
   
    return D_xz, D_xn, D_nz, S_xn, S_nz


# random negative sampling triplet loss
def triplet_loss_0(V, F, G, c):

    D_xz, D_xn, D_nz = random_sampling(F, G)
    loss = torch.clamp(D_xz - D_xn + c, min=0) + torch.clamp(D_xz - D_nz + c, min=0)
    return torch.sum(loss)


# hard negative sampling triplet loss
def triplet_loss_1(V, F, G, c):

    D_xz, D_xn, D_nz, _, _ = hard_sampling(V, F, G)
    loss = torch.clamp(D_xz - D_xn + c, min=0) + torch.clamp(D_xz - D_nz + c, min=0)
    return torch.sum(loss)


# hard negative sampling + soft weighted triplet loss
def triplet_loss_2(V, F, G, c):

    D_xz, D_xn, D_nz, S_xn, S_nz = hard_sampling(V, F, G)
    loss = S_xn * torch.clamp(D_xz - D_xn + c, min=0) + S_nz * torch.clamp(D_xz - D_nz + c, min=0)
    return torch.sum(loss)


# hard negative sampling + soft margin triplet loss
def triplet_loss_3(V, F, G, c):

    D_xz, D_xn, D_nz, S_xn, S_nz = hard_sampling(V, F, G)
    loss = torch.clamp(D_xz - D_xn + c * torch.log(1 + S_xn), min=0) + \
            torch.clamp(D_xz - D_nz + c * torch.log(1 + S_nz), min=0)
    return torch.sum(loss)
