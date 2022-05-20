from torch import nn


class LinearNet(nn.Module):
    def __init__(self, hidden_width=50, N=2, in_dim=0, out_dim=0, std_init=0.5):
        super().__init__()
        self.hidden_width = hidden_width
        self.N = N
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.std_init = std_init
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.in_dim, self.hidden_width, bias=False))
        for i in range(N - 2):
            self.layers.append(nn.Linear(self.hidden_width, self.hidden_width, bias=False))
        self.layers.append(nn.Linear(self.hidden_width, self.out_dim, bias=False))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.double()
            module.weight.data.normal_(mean=0.0, std=self.std_init)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def get_e2e_vec(self):
        W = self.layers[-1].weight.data
        for i in range(len(self.layers) - 2, -1, -1):
            W = W @ self.layers[i].weight.data
        return W
