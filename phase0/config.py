import torch


SEED = 42
BATCH_SIZE = 64
INPUT_SIZE = 28 * 28
HIDDEN_SIZE_1 = 512
HIDDEN_SIZE_2 = 256
OUTPUT_SIZE = 10
LEARNING_RATE = 1e-3
EPOCHS = 5
MODEL_PATH = "checkpoints/model.pth"
DEVICE = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else "cpu"
)
