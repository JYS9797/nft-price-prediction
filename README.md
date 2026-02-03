# NFT Price Prediction (Multimodal Classification)

This repository contains a lightweight, reproducible pipeline for **NFT price-level prediction**
using **text + image** features with optional **collection-level signals** (floor price / average price).

The code is intended for **security/system analysis research** (e.g., risk estimation and robustness checks)
and is **not** designed for automated trading or market manipulation.

## Model
- Text encoder: BERT (`bert-base-uncased`)
- Image encoder: ViT (`google/vit-base-patch16-224-in21k`)
- Fusion: attention-based fusion + collection features (floor/avg) → 3-class classifier

## Repo structure
- `scripts/prepare_data.py`: build train/val CSVs (paths, prices, labels)
- `scripts/train.py`: train and validate the model
- `scripts/eval.py`: evaluate a trained checkpoint and export metrics
- `src/`: reusable dataset/model utilities
- `configs/config.yaml`: paths + hyperparameters

## Data format
This repo expects a CSV with columns:
- `collection_address`, `token_id`
- `image_path`, `desc_path`
- `price`, `floor_price`, `avg_price`
- `label` (0/1/2)

> **Note:** actual images/descriptions are not included in this repository.

## Quickstart
1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Prepare data (example)
```bash
python scripts/prepare_data.py --root_dir nft_data_moralis --prices_csv nft_prices.csv --collections_csv collections_info_5000.csv
```

3) Train
```bash
python scripts/train.py --config configs/config.yaml
```

4) Evaluate
```bash
python scripts/eval.py --config configs/config.yaml --ckpt checkpoints/model.pth
```

## Ethics & scope
This project focuses on **prediction for analysis** and reproducible experimentation.
It does not provide tools for price manipulation or misuse.
