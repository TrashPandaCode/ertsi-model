from dataset import ReverbRoomDataset
from model import ReverbCNN
from torch.utils.data import DataLoader
from torch import nn, optim
import torch

params = {
    "epochs": 10,
    "batch_size": 32,
    "lr": 0.001,
    "freqs": [125, 250, 500, 1000, 2000, 4000],
    "model_out": "model.pth",
}

def train():
    device = torch.device("cpu")

    dataset = ReverbRoomDataset("data", freqs=params["freqs"])
    dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    model = ReverbCNN(num_frequencies=len(params["freqs"])).to(device)
    model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(params["epochs"]):
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
        print(f"Epoch [{epoch + 1}/{params['epochs']}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), params["model_out"])
    print(f"Model saved to {params['model_out']}")


if __name__ == "__main__":
    train()
