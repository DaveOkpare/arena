import torch


SEED = 42
BATCH_SIZE = 64
MODEL_PATH = "checkpoints/model.pth"
DEVICE = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else "cpu"
)
