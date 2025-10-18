import torch
import csv
import heapq
import os


def get_vote_probability(votes: dict, device=None):
    """
    votes: dict of class -> tensor([batch_size])
    Returns: tensor of shape (batch_size, num_classes)
    """
    # Ensure tensors are on the same device
    if device is None:
        device = next(iter(votes.values())).device

    # Stack into shape (batch_size, num_classes)
    vote_tensor = torch.stack([v.to(device).to(torch.float32) for v in votes.values()], dim=1)

    # Normalize to probabilities along classes
    total_votes = vote_tensor.sum(dim=1, keepdim=True)  # shape (batch_size, 1)
    vote_prob = vote_tensor / (total_votes + 1e-8)      # avoid div by zero

    return vote_prob


""" TRAIN UTILS """


def load_validation_history(val_log_file):
    """Load validation history from CSV file."""
    val_history = []
    if os.path.exists(val_log_file):
        with open(val_log_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                step, loss, path = int(row[0]), float(row[1]), row[2]
                val_history.append((step, loss, path))
    return val_history


def initialize_best_models_heap(val_history, k=5):
    """Initialize heap of best models from validation history."""
    best_models = [(-loss, path) for _, loss, path in val_history]
    heapq.heapify(best_models)
    best_models = heapq.nsmallest(k, best_models)
    heapq.heapify(best_models)
    return best_models


def save_checkpoint(checkpoint_path, epoch, step, model, spectral_conv, optimizer, scaler, val_loss=None):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "spectral_convolution": spectral_conv.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if val_loss is not None:
        checkpoint["val_loss"] = val_loss
    
    torch.save(checkpoint, checkpoint_path)


def log_validation_loss(val_log_file, step, val_loss, save_path):
    """Log validation loss to CSV file."""
    write_header = not os.path.exists(val_log_file)
    with open(val_log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "val_loss", "checkpoint_path"])
        writer.writerow([step, val_loss, save_path])


def maintain_best_models(best_models, val_loss, save_path, max_models=5):
    """Maintain top-k best models, removing worst checkpoint if needed."""
    heapq.heappush(best_models, (-val_loss, save_path))
    if len(best_models) > max_models:
        worst = heapq.heappop(best_models)
        try:
            os.remove(worst[1])
            print(f"Removed old checkpoint: {worst[1]}")
        except OSError:
            pass
