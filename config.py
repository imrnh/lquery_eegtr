import torch


config = {
    'output_dim': 6,
    'model_embed_dim': 512,
    'num_layers': 3,
    'num_heads': 8,
    'd_ff': 2048, # 4 * d_model
    'num_learnable_queries': 8, # Number of special query tokens
    'dropout': 0.2,  # Dropout for standard self-attention
    'dropout_mlp': 0.1,
    'dropout_embed': 0.1,
    'eeg_channels': 26,
    'timestamp': 99,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_dir = ["data/train"]
val_dir = ["data/val"]