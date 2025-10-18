import torch
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
    prefetch_factor=2
)



# Data processor
train_ds = EEGDataset(train_dir, metadata_file="lib/train_metadata.pkl")
val_ds = EEGDataset(val_dir, metadata_file="lib/val_metadata.pkl")

dataloader = DataLoader(train_ds, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_dataloader = DataLoader(val_ds, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)

model = Transformer(**config).to(device)
model.train() # Set model to training mode to test dropout
print(f"Model created on {device} with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")