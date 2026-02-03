import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTImageProcessor
from PIL import Image, ImageFile
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")


class NFTDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.floor_min = self.data["floor_price"].min()
        self.floor_max = self.data["floor_price"].max()
        self.avg_min = self.data["avg_price"].min()
        self.avg_max = self.data["avg_price"].max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image_inputs = image_processor(images=image, return_tensors="pt")

        with open(row["desc_path"], "r", encoding="utf-8") as f:
            description = f.read().strip()
        text_inputs = tokenizer(
            description,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        floor_price = (row["floor_price"] - self.floor_min) / (
            self.floor_max - self.floor_min
        )
        avg_price = (row["avg_price"] - self.avg_min) / (self.avg_max - self.avg_min)

        floor_price = (
            torch.tensor(floor_price, dtype=torch.float).unsqueeze(0).to(device)
        )
        avg_price = torch.tensor(avg_price, dtype=torch.float).unsqueeze(0).to(device)

        # floor_price = (
        #     torch.tensor(row["floor_price"], dtype=torch.float).unsqueeze(0).to(device)
        # )
        # avg_price = (
        #     torch.tensor(row["avg_price"], dtype=torch.float).unsqueeze(0).to(device)
        # )

        return (
            text_inputs["input_ids"].squeeze(0).to(device),
            text_inputs["attention_mask"].squeeze(0).to(device),
            image_inputs["pixel_values"].squeeze(0).to(device),
            floor_price,
            avg_price,
            torch.tensor(row["label"], dtype=torch.long).to(device),
        )
