from dataset import ReverbDataset
from model import ReverbCNN
from torch.utils.data import DataLoader
from torch import nn
import torch
import json

params = {
    "epochs": 10,
    "batch_size": 32,
    "lr": 0.001,
    "freq_subset": [125, 250, 500, 1000, 2000, 4000],
    "model_out": "model.pth",
}

def evaluate():
    cfg = params["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ReverbDataset(cfg["data_root"], freqs=cfg["freqs"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = ReverbCNN(num_frequencies=len(cfg["freqs"]))
    model.load_state_dict(torch.load(cfg["model_out"]))
    model.to(device)
    model = torch.compile(model)
    model.eval()

    loss_fn = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f"MSE: {avg_loss:.4f}")

    with open("eval_metrics.json", "w") as f:
        json.dump({"mse": avg_loss}, f)

if __name__ == "__main__":
    evaluate()