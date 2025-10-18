import torch.nn as nn

class MLP(nn.Module):
    """
    A standard Feed-Forward Network (MLP) used in Transformer encoders.
    It consists of two linear layers with a GELU activation in between.
    """
    def __init__(self, d_model, d_ff, dropout):
        """
        Args:
            d_model (int): The dimension of the input and output.
            d_ff (int): The dimension of the hidden layer.
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
