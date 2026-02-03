\
import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import NFTDataset
from src.models.merlin_classifier import MERLINClassifier
from src.training.utils import set_seed, ensure_dir


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train NFT price-level classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = cfg["data"]["root_dir"]
    train_csv = os.path.join(root_dir, cfg["data"]["train_csv"])
    val_csv = os.path.join(root_dir, cfg["data"]["val_csv"])

    train_ds = NFTDataset(
        train_csv,
        max_text_len=int(cfg["model"]["max_text_len"]),
        text_model_name=cfg["model"]["text_model"],
        image_model_name=cfg["model"]["image_model"],
    )
    val_ds = NFTDataset(
        val_csv,
        max_text_len=int(cfg["model"]["max_text_len"]),
        text_model_name=cfg["model"]["text_model"],
        image_model_name=cfg["model"]["image_model"],
    )

    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)

    model = MERLINClassifier(
        text_model_name=cfg["model"]["text_model"],
        image_model_name=cfg["model"]["image_model"],
        num_classes=int(cfg["model"]["num_classes"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    epochs = int(cfg["train"]["epochs"])

    ensure_dir(cfg["output"]["checkpoints_dir"])
    best_path = os.path.join(cfg["output"]["checkpoints_dir"], "model.pth")

    best_val_acc = -1.0
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            input_ids, attention_mask, pixel_values, floor_price, avg_price, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pixel_values = pixel_values.to(device)
            floor_price = floor_price.to(device)
            avg_price = avg_price.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, pixel_values, floor_price, avg_price)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            correct += int((logits.argmax(1) == labels).sum().item())

        train_acc = correct / len(train_loader.dataset)
        train_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                input_ids, attention_mask, pixel_values, floor_price, avg_price, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                pixel_values = pixel_values.to(device)
                floor_price = floor_price.to(device)
                avg_price = avg_price.to(device)
                labels = labels.to(device)

                logits = model(input_ids, attention_mask, pixel_values, floor_price, avg_price)
                loss = criterion(logits, labels)
                val_loss += float(loss.item())
                val_correct += int((logits.argmax(1) == labels).sum().item())

        val_acc = val_correct / len(val_loader.dataset)
        val_loss = val_loss / max(len(val_loader), 1)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved best checkpoint to: {best_path} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
