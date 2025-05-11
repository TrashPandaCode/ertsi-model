from dataset import ReverbDataset
from model import ReverbCNN
from torch.utils.data import DataLoader
from torch import nn, optim
import yaml
import torch

def load_params(path="params.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train():
    params = load_params()
    cfg = params["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ReverbDataset(cfg["data_root"], freqs=cfg["freqs"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = ReverbCNN(num_frequencies=len(cfg["freqs"])).to(device)
    model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(cfg["epochs"]):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{cfg['epochs']}], Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), cfg["model_out"])
    print(f"Model saved to {cfg['model_out']}")

if __name__ == "__main__":
    train()