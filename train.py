# train.py
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from dataset import IRDataset
from model import IRClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = IRDataset("metadata.json")
train_len = int(0.8 * len(dataset))
train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32)

model = IRClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, angles, times, labels in train_loader:
        imgs, angles, times, labels = imgs.to(device), angles.to(device), times.to(device), labels.to(device)
        preds = model(imgs, angles, times)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "model.pt")
