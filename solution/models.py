import torch


class ProjectionModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # projection to latent subspace
        self.project = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # project to latent subspace
        F = self.project(x)
        # normalize
        F = torch.nn.functional.normalize(F)

        return F