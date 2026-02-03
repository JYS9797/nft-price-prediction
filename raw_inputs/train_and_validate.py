import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nft_dataset import NFTDataset
from merlin_classification import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "nft_data_moralis/"
train_dataset = NFTDataset(root_dir + "train_nft.csv")
val_dataset = NFTDataset(root_dir + "val_nft.csv")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0

    for (
        input_ids,
        attention_mask,
        pixel_values,
        floor_price,
        avg_price,
        labels,
    ) in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, pixel_values, floor_price, avg_price)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_loader.dataset)
    train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for (
            input_ids,
            attention_mask,
            pixel_values,
            floor_price,
            avg_price,
            labels,
        ) in val_loader:
            outputs = model(
                input_ids, attention_mask, pixel_values, floor_price, avg_price
            )
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_loader.dataset)
    val_loss = val_loss / len(val_loader)

    print(
        f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

torch.save(model.state_dict(), "merlin_classifier_final_with_collection.pth")
