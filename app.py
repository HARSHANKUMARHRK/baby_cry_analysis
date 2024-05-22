import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder

from torchvision.transforms import Resize



# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(3 * 640 * 480, 256),  # Assuming images are RGB with size 640x480
            nn.ReLU(),
            nn.Linear(256, 5)  # 5 output classes
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.flatten(input_data)
        logits = self.dense_layers(x)
        predictions = self.softmax(logits)
        return predictions




# Function to train a single epoch
def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Training Loss: {loss.item()}")


# Function to train the model
def train(model, train_loader, loss_fn, optimizer, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}:")
        train_single_epoch(model, train_loader, loss_fn, optimizer, device)
        print("---------------------------")
    print("Training finished.")


if __name__ == "__main__":
    # Define your data directories
    root_dir = "output"

    # Set hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Create custom dataset and dataloader
    custom_dataset = CustomDataset(root_dir=root_dir, transform=ToTensor())
    train_dataloader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Construct model
    model = FeedForwardNet()

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, train_dataloader, loss_fn, optimizer, device, EPOCHS)

    # Save the trained model
    torch.save(model.state_dict(), "baby.pth")
    print("Trained model saved.")
