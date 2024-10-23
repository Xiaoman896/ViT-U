# Vision Transformer (ViT) fills gaps in the sinogram domain

## Create a New Environment from the YAML File
conda env create -f environment.yaml
## Train a ViT model
python main-ViT.py

Before training, checking your configurations are correct (config/ViT.yaml)
## Dataset (h5):
For demonstration, a small simulated dataset (sinogram data) can be downloaded from [here](https://drive.google.com/drive/folders/19BIugC-aL9Ijpk8WWb_XWZW2X3A15Xgr)

## Reference
    The torch implementation of ViT (https://github.com/lzhengchun/TomoTx)

