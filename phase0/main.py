import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    HIDDEN_SIZE,
    INPUT_SIZE,
    LEARNING_RATE,
    MODEL_PATH,
    OUTPUT_SIZE,
    SEED,
)
from model import MLP
from utils import set_seed

set_seed(SEED)

train_dataset = datasets.mnist.MNIST(
    root="data", download=True, train=True, transform=ToTensor()
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

model = MLP(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
model = model.to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    import os

    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("Done!")
