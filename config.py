import torch


config = {
    'output_dim': 512,
    'model_embed_dim': 512,
    'num_layers': 5,
    'num_heads': 8,
    'd_ff': 2048, # 4 * d_model
    'num_learnable_queries': 8, # Number of special query tokens
    'dropout_attn_xp': 0.3,  # Dropout for standard self-attention
    'dropout_attn_lqp': 0.1, # Higher dropout for attention involving queries
    'dropout_mlp': 0.1,
    'dropout_embed': 0.1,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
