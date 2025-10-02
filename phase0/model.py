from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_in = nn.Linear(input_size, hidden_size_1)
        self.relu = nn.ReLU()
        self.linear_mid = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear_out = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_in(x)
        out = self.relu(out)
        out = self.linear_mid(out)
        out = self.relu(out)
        out = self.linear_out(out)
        return out
