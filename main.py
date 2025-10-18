import torch
from convolution import FrequencyBinConvolution
from data_processor import EEGDataset
from model.transformer import Transformer
from config import config, device, train_dir, val_dir

from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat

from types import SimpleNamespace

hyperparams = SimpleNamespace(
    batch_size=12,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    learning_rate=1e-3,
    num_epochs = 10,
)



# Data processor
train_ds = EEGDataset(train_dir, metadata_file="lib/train_metadata.pkl")
val_ds = EEGDataset(val_dir, metadata_file="lib/val_metadata.pkl")

dataloader = DataLoader(train_ds, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_dataloader = DataLoader(val_ds, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)


# Frequency bin convolution
freq_bin_conv = FrequencyBinConvolution(embed_dim=config['model_embed_dim']).to(device)


# Model
model = Transformer(**config).to(device)
model.train() # Set model to training mode to test dropout
print(f"Model created on {device} with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")


# Optimizer and Loss
optimizer = torch.optim.AdamW(list(model.parameters()) + list(freq_bin_conv.parameters()),lr = hyperparams.learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(hyperparams.num_epochs):
    for batch in dataloader:
        pass

