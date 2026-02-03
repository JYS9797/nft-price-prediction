## Pipeline overview

1) `scripts/prepare_data.py`
- merges per-NFT prices with collection-level statistics
- validates presence of image/text assets
- creates 3-class labels using Q1/Q3 quantiles
- writes train/val CSV splits

2) `scripts/train.py`
- trains MERLINClassifier (BERT + ViT + collection features)
- saves best checkpoint to `checkpoints/model.pth`

3) `scripts/eval.py`
- evaluates checkpoint on validation split
- exports metrics to `outputs/evaluation_results.csv`
