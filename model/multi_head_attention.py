import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    A simplified Multi-Head Attention module with standard dropout.
    """
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = dropout

    def forward(self, x, num_x_tokens=None):
        """
        Forward pass for Multi-Head Attention.

        Args:
            x (torch.Tensor): Input tensor of shape
                              (batch_size, timestamps, num_tokens, embedding_dim).
            num_x_tokens (int, optional): Kept for API compatibility but not used.
        """
        b, timestamps, num_tokens, d_in = x.shape

        # Compute queries, keys, and values
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape and transpose for multi-head processing
        keys = keys.view(b, timestamps, num_tokens, self.num_heads, self.head_dim).transpose(2, 3)
        queries = queries.view(b, timestamps, num_tokens, self.num_heads, self.head_dim).transpose(2, 3)
        values = values.view(b, timestamps, num_tokens, self.num_heads, self.head_dim).transpose(2, 3)
        
        # Compute attention scores and weights
        attn_scores = queries @ keys.transpose(-1, -2)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        # Apply standard dropout to attention weights
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute context vector
        context_vec = (attn_weights @ values).transpose(2, 3)
        context_vec = context_vec.contiguous().view(b, timestamps, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec