import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from config import BATCH_SIZE, DEVICE, MODEL_PATH, SEED
from utils import set_seed

set_seed(SEED)

test_dataset = datasets.mnist.MNIST(root="data", train=False, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


def eval(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )


if __name__ == "__main__":
    from model import MLP
    from config import INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE

    _model = MLP(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE)
    _model.load_state_dict(torch.load(MODEL_PATH))
    _model.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    eval(test_loader, _model, loss_fn)
