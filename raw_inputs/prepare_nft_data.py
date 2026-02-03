import os
import pandas as pd
import imghdr
from sklearn.model_selection import train_test_split

root_dir = "nft_data_moralis/"
price_csv = os.path.join(root_dir, "nft_prices.csv")
output_csv = os.path.join(root_dir, "valid_nft_dataset_classification.csv")

price_df = pd.read_csv(price_csv)
collections_info = pd.read_csv("/home/isslab/Desktop/nft/ver3_customized merlin/collections_info_5000.csv")
collections_info = collections_info.rename(
    columns={
        "tokenAddress": "collection_address",
        "AveragePriceRecent": "avg_price",
        "FloorPriceRecent": "floor_price",
    }
)

merged_df = price_df.merge(
    collections_info[["collection_address", "floor_price", "avg_price"]],
    on="collection_address",
    how="left",
)

merged_df = merged_df.dropna(subset=["floor_price", "avg_price"])

valid_data = []
error_log_path = os.path.join(root_dir, "error_log.txt")

for _, row in merged_df.iterrows():
    collection_address, token_id = row["collection_address"], row["token_id"]
    price, floor, avg = row["price"], row["floor_price"], row["avg_price"]
    image_path = os.path.join(root_dir, collection_address, f"{token_id}_image.jpg")
    desc_path = os.path.join(
        root_dir, collection_address, f"{token_id}_description.txt"
    )

    if (
        os.path.isfile(image_path)
        and os.path.isfile(desc_path)
        and imghdr.what(image_path) is not None
    ):
        valid_data.append(
            {
                "collection_address": collection_address,
                "token_id": token_id,
                "image_path": image_path,
                "desc_path": desc_path,
                "price": price,
                "floor_price": floor,
                "avg_price": avg,
            }
        )
    else:
        with open(error_log_path, "a") as log_file:
            log_file.write(
                f"Missing or invalid files for {collection_address}/{token_id}\n"
            )

valid_df = pd.DataFrame(valid_data)
valid_df["price"] = pd.to_numeric(valid_df["price"], errors="coerce")
valid_df = valid_df.dropna(subset=["price"])
q1 = valid_df["price"].quantile(0.25)
q3 = valid_df["price"].quantile(0.75)
print(f"Q1: {q1}, Q3: {q3}")
# save quantiles for later use
quantiles_path = "quantiles.txt"
with open(quantiles_path, "w") as f:
    f.write(f"Q1: {q1}\nQ3: {q3}\n")


def label_price(p):
    if p <= q1:
        return 0
    elif p <= q3:
        return 1
    else:
        return 2


valid_df["label"] = valid_df["price"].apply(label_price)

# Split into train and validation
train_df, val_df = train_test_split(
    valid_df, test_size=0.1, random_state=42, stratify=valid_df["label"]
)

# Save splits
train_df.to_csv(os.path.join(root_dir, "train_nft.csv"), index=False)
val_df.to_csv(os.path.join(root_dir, "val_nft.csv"), index=False)
print("Data preparation complete. Train/validation data saved.")
