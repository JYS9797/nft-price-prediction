\
import argparse
import os
import time
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from src.data.dataset import NFTDataset
from src.models.merlin_classifier import MERLINClassifier


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained NFT price classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth).")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = cfg["data"]["root_dir"]
    val_csv = os.path.join(root_dir, cfg["data"]["val_csv"])

    dataset = NFTDataset(
        val_csv,
        max_text_len=int(cfg["model"]["max_text_len"]),
        text_model_name=cfg["model"]["text_model"],
        image_model_name=cfg["model"]["image_model"],
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = MERLINClassifier(
        text_model_name=cfg["model"]["text_model"],
        image_model_name=cfg["model"]["image_model"],
        num_classes=int(cfg["model"]["num_classes"]),
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    true_labels, predictions, inference_times = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, pixel_values, floor_price, avg_price, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pixel_values = pixel_values.to(device)
            floor_price = floor_price.to(device)
            avg_price = avg_price.to(device)
            labels = labels.to(device)

            t0 = time.time()
            outputs = model(input_ids, attention_mask, pixel_values, floor_price, avg_price)
            t1 = time.time()
            inference_times.append(t1 - t0)

            preds = torch.argmax(outputs, dim=1)
            true_labels.extend(labels.cpu().numpy().tolist())
            predictions.extend(preds.cpu().numpy().tolist())

    accuracy = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)

    # Metrics aligned with your customized evaluation
    wr = cm[2, 2] / (cm[:, 2].sum() + 1e-12)
    lr = (cm[0, 2] + cm[1, 2]) / (cm[:, 2].sum() + 1e-12)
    wlr = wr / (lr + 1e-12)
    mr = (cm[2, 0] + cm[2, 1]) / (cm[2, :].sum() + 1e-12)
    cn = mr / ((mr + lr) + 1e-12)
    rn = lr / ((mr + lr) + 1e-12)

    inference_times = np.array(inference_times)
    speed_stats = {
        "mean_time": float(inference_times.mean()) if len(inference_times) else 0.0,
        "max_time": float(inference_times.max()) if len(inference_times) else 0.0,
        "min_time": float(inference_times.min()) if len(inference_times) else 0.0,
        "std_time": float(inference_times.std()) if len(inference_times) else 0.0,
    }

    results = {
        "accuracy": float(accuracy),
        "wr": float(wr),
        "lr": float(lr),
        "wlr": float(wlr),
        "mr": float(mr),
        "cn": float(cn),
        "rn": float(rn),
        **speed_stats,
    }

    os.makedirs(cfg["output"]["metrics_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output"]["metrics_dir"], "evaluation_results.csv")
    pd.DataFrame([results]).to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(pd.DataFrame([results]))


if __name__ == "__main__":
    main()
