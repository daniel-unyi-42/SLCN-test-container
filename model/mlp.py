import torch

from torch_geometric.nn import global_max_pool, global_mean_pool
from torch.nn import Module, Linear, BatchNorm1d, Dropout
from torch.nn.functional import relu, selu
from torch import sigmoid, tanh

class MLP(Module):

    def __init__(self, in_dim, hidden_dims, out_dim, device):
        super(MLP, self).__init__()
        self.device = device
        self.lin1 = Linear(in_dim, hidden_dims[0])
        self.lin2 = Linear(hidden_dims[0], hidden_dims[1])
        self.lin3 = Linear(hidden_dims[1], hidden_dims[2])
        self.lin4 = Linear(hidden_dims[2], hidden_dims[3])
        self.lin5 = Linear(hidden_dims[3] * 2, hidden_dims[3])
        self.lin6 = Linear(hidden_dims[3], out_dim)
        self.to(device)

    def forward(self, data):
        data = data.to(self.device)
        batch = data.batch
        x = data.x
        x = tanh(self.lin1(x))
        x = tanh(self.lin2(x))
        x = tanh(self.lin3(x))
        x = tanh(self.lin4(x))
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_c = torch.cat([x_max, x_mean], dim=1)
        x_c = tanh(self.lin5(x_c))
        return self.lin6(x_c)
