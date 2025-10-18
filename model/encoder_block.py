import torch.nn as nn
from model.multi_head_attention import MultiHeadAttention
from model.mlp import MLP

class EncoderBlock(nn.Module):
    """
    A single Transformer Encoder block.

    This block consists of a Multi-Head Attention layer followed by a Feed-Forward
    Network (MLP). It includes layer normalization before each sub-layer and
    residual connections after each sub-layer.
    """
    def __init__(self, model_embed_dim, num_heads, d_ff, dropout_attn, dropout_attn_xp, dropout_attn_lqp, dropout_mlp):
        """
        Args:
            model_embed_dim (int): The dimension of the input, attention, and MLP.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimension of the MLP's hidden layer.
            dropout_attn (float): Dropout for the final attention projection.
                                 (Note: this is not currently used in the custom MHA but is standard)
            dropout_attn_xp (float): Dropout for the x-to-x part of attention weights.
            dropout_attn_lqp (float): Dropout for parts of attention weights involving learnable queries.
            dropout_mlp (float): Dropout for the MLP layer.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(model_embed_dim)
        self.norm2 = nn.LayerNorm(model_embed_dim)
        self.attention = MultiHeadAttention(
            d_in=model_embed_dim,
            d_out=model_embed_dim,
            dropout_xp=dropout_attn_xp,
            dropout_lqp=dropout_attn_lqp,
            num_heads=num_heads
        )
        self.mlp = MLP(model_embed_dim, d_ff, dropout_mlp)
        

    def forward(self, x, num_x_tokens):
        """
        Forward pass for the Encoder Block.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, total_tokens, model_embed_dim).
            num_x_tokens (int): The number of original tokens, passed to the attention layer.
        """
        # --- Attention Sub-layer with Residual Connection ---
        # x + Attention(LayerNorm(x))
        attn_output = self.attention(self.norm1(x), num_x_tokens=num_x_tokens)
        x = x + attn_output

        # --- MLP Sub-layer with Residual Connection ---
        # x + MLP(LayerNorm(x))
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output

        return x
