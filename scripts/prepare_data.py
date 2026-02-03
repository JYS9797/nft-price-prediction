\
import os
import argparse
import imghdr
import pandas as pd
from sklearn.model_selection import train_test_split


def label_price(price: float, q1: float, q3: float) -> int:
    if price <= q1:
        return 0
    elif price <= q3:
        return 1
    return 2


def main():
    parser = argparse.ArgumentParser(description="Prepare NFT dataset CSVs for multimodal price classification.")
    parser.add_argument("--root_dir", required=True, help="Root directory containing NFT images/descriptions.")
    parser.add_argument("--prices_csv", required=True, help="CSV with columns: collection_address, token_id, price.")
    parser.add_argument("--collections_csv", required=True, help="CSV with columns: tokenAddress, FloorPriceRecent, AveragePriceRecent.")
    parser.add_argument("--out_valid_csv", default="valid_nft_dataset_classification.csv")
    parser.add_argument("--out_train_csv", default="train_nft.csv")
    parser.add_argument("--out_val_csv", default="val_nft.csv")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    price_df = pd.read_csv(args.prices_csv)
    collections_info = pd.read_csv(args.collections_csv).rename(
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
    ).dropna(subset=["floor_price", "avg_price"])

    valid = []
    error_log_path = os.path.join(args.root_dir, "error_log.txt")

    for _, row in merged_df.iterrows():
        collection_address, token_id = row["collection_address"], row["token_id"]
        price, floor, avg = row["price"], row["floor_price"], row["avg_price"]

        image_path = os.path.join(args.root_dir, collection_address, f"{token_id}_image.jpg")
        desc_path = os.path.join(args.root_dir, collection_address, f"{token_id}_description.txt")

        ok = (
            os.path.isfile(image_path)
            and os.path.isfile(desc_path)
            and imghdr.what(image_path) is not None
        )
        if ok:
            valid.append(
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
            with open(error_log_path, "a", encoding="utf-8") as log:
                log.write(f"Missing or invalid files for {collection_address}/{token_id}\n")

    valid_df = pd.DataFrame(valid)
    valid_df["price"] = pd.to_numeric(valid_df["price"], errors="coerce")
    valid_df = valid_df.dropna(subset=["price"])

    q1 = valid_df["price"].quantile(0.25)
    q3 = valid_df["price"].quantile(0.75)

    # save quantiles for reference
    with open(os.path.join(args.root_dir, "quantiles.txt"), "w", encoding="utf-8") as f:
        f.write(f"Q1: {q1}\nQ3: {q3}\n")

    valid_df["label"] = valid_df["price"].apply(lambda p: label_price(float(p), q1, q3))

    train_df, val_df = train_test_split(
        valid_df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=valid_df["label"],
    )

    valid_df.to_csv(os.path.join(args.root_dir, args.out_valid_csv), index=False)
    train_df.to_csv(os.path.join(args.root_dir, args.out_train_csv), index=False)
    val_df.to_csv(os.path.join(args.root_dir, args.out_val_csv), index=False)

    print("Data preparation complete.")
    print(f"- valid: {len(valid_df)} rows")
    print(f"- train: {len(train_df)} rows")
    print(f"- val  : {len(val_df)} rows")
    print(f"Saved to: {args.root_dir}")


if __name__ == "__main__":
    main()
