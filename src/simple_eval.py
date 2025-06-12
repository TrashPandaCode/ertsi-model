import torch
from torch.utils.data import DataLoader
from simple_model import SimpleReverbCNN
from dataset import ReverbRoomDataset
from sklearn.metrics import mean_squared_error
import numpy as np


def evaluate():
    freqs = [250, 500, 1000, 2000, 4000, 8000]

    dataset = ReverbRoomDataset("data/test/real", freqs=freqs, augment=False)
    loader = DataLoader(dataset, batch_size=32, num_workers=4)

    model = SimpleReverbCNN.load_from_checkpoint("output/simple_reverbcnn.ckpt")
    model.eval()
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
            targets.append(y.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mse = mean_squared_error(targets, preds)
    print(f"MSE: {mse:.4f}")


if __name__ == "__main__":
    evaluate()
