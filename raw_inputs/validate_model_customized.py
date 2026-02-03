import torch
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, ViTImageProcessor
from PIL import Image
from merlin_classification import MERLINClassifier, device
from sklearn.metrics import accuracy_score, confusion_matrix


# Custom dataset
class NFTDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.image_processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.floor_min = self.data["floor_price"].min()
        self.floor_max = self.data["floor_price"].max()
        self.avg_min = self.data["avg_price"].min()
        self.avg_max = self.data["avg_price"].max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        with open(row["desc_path"], "r", encoding="utf-8") as f:
            text = f.read().strip()

        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values
        floor_price = (row["floor_price"] - self.floor_min) / (
            self.floor_max - self.floor_min
        )
        avg_price = (row["avg_price"] - self.avg_min) / (self.avg_max - self.avg_min)
        floor_price = torch.tensor([floor_price], dtype=torch.float32)
        avg_price = torch.tensor([avg_price], dtype=torch.float32)
        label = row["label"]
        return (
            inputs["input_ids"].squeeze(0),
            inputs["attention_mask"].squeeze(0),
            pixel_values.squeeze(0),
            floor_price,
            avg_price,
            label,
        )


# Load dataset
val_df = pd.read_csv("E:/merlin/nft_data_moralis/val_nft.csv")
dataset = NFTDataset(val_df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load model
model = MERLINClassifier().to(device)
model.load_state_dict(torch.load("merlin_classifier_final_with_collection.pth"))
model.eval()

# Metrics storage
true_labels, predictions = [], []
inference_times = []

# Inference
with torch.no_grad():
    for (
        input_ids,
        attention_mask,
        pixel_values,
        floor_price,
        avg_price,
        labels,
    ) in dataloader:
        input_ids, attention_mask, pixel_values = (
            input_ids.to(device),
            attention_mask.to(device),
            pixel_values.to(device),
        )
        floor_price, avg_price = floor_price.to(device), avg_price.to(device)
        start_time = time.time()
        outputs = model(input_ids, attention_mask, pixel_values, floor_price, avg_price)
        end_time = time.time()
        inference_times.append(end_time - start_time)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        true_labels.extend(labels.numpy())
        predictions.extend(preds)

# Metrics calculation
accuracy = accuracy_score(true_labels, predictions)
cm = confusion_matrix(true_labels, predictions)

# Metrics from the provided paper
wr = cm[2, 2] / (cm[:, 2].sum())
lr = (cm[0, 2] + cm[1, 2]) / cm[:, 2].sum()
wlr = wr / lr if lr != 0 else np.inf
mr = (cm[2, 0] + cm[2, 1]) / cm[2, :].sum()
cn = mr / (mr + lr) if (mr + lr) != 0 else 0
rn = lr / (mr + lr) if (mr + lr) != 0 else 0

# Inference speed stats
inference_times = np.array(inference_times)
speed_stats = {
    "mean_time": inference_times.mean(),
    "max_time": inference_times.max(),
    "min_time": inference_times.min(),
    "std_time": inference_times.std(),
}

# Store results
results = {
    "accuracy": accuracy,
    "wr": wr,
    "lr": lr,
    "wlr": wlr,
    "mr": mr,
    "cn": cn,
    "rn": rn,
    **speed_stats,
}

# Save results
results_df = pd.DataFrame([results])
results_df.to_csv("evaluation_results.csv", index=False)

# Print results
print(results_df)
