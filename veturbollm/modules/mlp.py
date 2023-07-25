import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        bias1=True,
        bias2=True,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, device=device, dtype=dtype)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, device=device, dtype=dtype)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

    def reset_parameters(self, initializer_range=0.02):
        nn.init.normal_(self.fc1.weight, std=initializer_range)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=initializer_range)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)


class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.sigmoid,
        bias1=True,
        bias2=True,
        multiple_of=256,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(8 * in_features / 3)
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias1, device=device, dtype=dtype)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias1, device=device, dtype=dtype)

    def forward(self, x):
        y = self.fc1(x)
        if self.activation == F.sigmoid:  # Special case for GLU
            y = F.glu(y, dim=-1)
        else:
            y, gate = y.chunk(2, dim=-1)
            y = y * self.activation(gate)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

    def reset_parameters(self, initializer_range=0.02):
        nn.init.normal_(self.fc1.weight, std=initializer_range)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=initializer_range)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
