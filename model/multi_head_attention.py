import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):    
    def __init__(self, d_in, d_out, context_length, dropout_xp, dropout_lqp, num_heads, qkv_bias=False):
        super().__init__()
        
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        # We store the probabilities, not the modules,
        # as we will be applying the dropout logic manually.
        self.dropout_p_x = dropout_xp
        self.dropout_p_lq = dropout_lqp

    def forward(self, x, learnable_queries=None):
        b, num_x_tokens, d_in = x.shape

        if learnable_queries is not None:
            b_lq, num_lq_tokens, d_in_lq = learnable_queries.shape
            assert b == b_lq, "Batch dimensions must match"
            assert d_in == d_in_lq, "Input dimensions must match"
            
            combined_input = torch.cat([x, learnable_queries], dim=1)
            total_tokens = num_x_tokens + num_lq_tokens
        else:
            combined_input = x
            num_lq_tokens = 0
            total_tokens = num_x_tokens

        keys = self.W_key(combined_input)
        queries = self.W_query(combined_input)
        values = self.W_value(combined_input)

        keys = keys.view(b, total_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, total_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, total_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(-1, -2)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        
        # Dropout is only applied during training
        if self.training:
            # Get keep probabilities
            keep_p_x = 1.0 - self.dropout_p_x
            keep_p_lq = 1.0 - self.dropout_p_lq
            
            if learnable_queries is not None:
                # Split the attention matrix
                weights_xx = attn_weights[:, :, :num_x_tokens, :num_x_tokens]
                weights_xlq = attn_weights[:, :, :num_x_tokens, num_x_tokens:]
                weights_lqx = attn_weights[:, :, num_x_tokens:, :num_x_tokens]
                weights_lqlq = attn_weights[:, :, num_x_tokens:, num_x_tokens:]

                # Create binary masks (no scaling yet)
                # We use torch.rand_like and compare to the *keep* probability
                mask_xx = (torch.rand_like(weights_xx) < keep_p_x).float()
                mask_xlq = (torch.rand_like(weights_xlq) < keep_p_lq).float()
                mask_lqx = (torch.rand_like(weights_lqx) < keep_p_lq).float()
                mask_lqlq = (torch.rand_like(weights_lqlq) < keep_p_lq).float()

                # Apply masks
                masked_xx = weights_xx * mask_xx
                masked_xlq = weights_xlq * mask_xlq
                masked_lqx = weights_lqx * mask_lqx
                masked_lqlq = weights_lqlq * mask_lqlq
                
                # Recombine the matrix
                top_row = torch.cat([masked_xx, masked_xlq], dim=-1)
                bottom_row = torch.cat([masked_lqx, masked_lqlq], dim=-1)
                masked_attn_weights = torch.cat([top_row, bottom_row], dim=-2)
                
                # Calculate the combined scaling factor i.e. number of elements in each block
                n_xx = num_x_tokens * num_x_tokens
                n_xlq = num_x_tokens * num_lq_tokens
                n_lqx = num_lq_tokens * num_x_tokens
                n_lqlq = num_lq_tokens * num_lq_tokens
                
                n_lq_total = n_xlq + n_lqx + n_lqlq
                n_total = n_xx + n_lq_total

                # Expected number of elements kept
                expected_kept_xx = n_xx * keep_p_x
                expected_kept_lq = n_lq_total * keep_p_lq
                expected_total_kept = expected_kept_xx + expected_kept_lq
                
                # The combined scaling factor is TotalElements / ExpectedKeptElements
                if expected_total_kept > 0:
                    scale = n_total / expected_total_kept
                else:
                    scale = 0.0 # All elements were dropped
                
                # 6. Apply the single, combined scaling factor
                attn_weights = masked_attn_weights * scale

            else:
                # Original behavior: only dropout_x, but applied manually
                # F.dropout applies scaling, so we do it manually to be consistent
                keep_p_x = 1.0 - self.dropout_p_x
                mask_xx = (torch.rand_like(attn_weights) < keep_p_x).float()
                masked_attn_weights = attn_weights * mask_xx
                
                # Scale is just 1 / keep_p_x
                if keep_p_x > 0:
                    scale = 1.0 / keep_p_x
                else:
                    scale = 0.0
                
                attn_weights = masked_attn_weights * scale
        
        # During eval (self.training=False), no dropout is applied,
        # so we just use the original attn_weights
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, total_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec