import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
from pali3.main import VitModel
from zeta.nn.modules import SigLipLoss


def pretrain(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    scheduler,
    num_epochs,
    model_path,
):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            embeddings = model.process(images)
            loss = loss_fn(embeddings, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_loss = validate(model, val_loader, loss_fn, device)
        print(f"Epoch: {epoch+1}, Train Loss: {loss.item()}, Val Loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)


def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model.process(images)
            loss = loss_fn(embeddings, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# Dataset and DataLoader
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
val_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# Model, Optimizer, Scheduler and Loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VitModel(image_size=32, patch_size=4, dim=512, depth=6, heads=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = SigLipLoss()

# Pretrain the model
model_path = "./model.pth"
num_epochs = 100
pretrain(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    scheduler,
    num_epochs,
    model_path,
)
