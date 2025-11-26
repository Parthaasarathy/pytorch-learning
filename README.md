# pytorch-learning

A repository to store PyTorch learning code, models, experiments, and notes.

## Folder Structure
- `scripts/` — Python training and inference scripts  
- `models/` — saved model checkpoints (`*.pth`)  
- `notes/` — Markdown notes  
- `notebooks/` — Jupyter notebooks  

## Quick Start

### Create environment
```bash
conda create -n torch-env python=3.11 -y
conda activate torch-env
pip install torch torchvision matplotlib
python3 scripts/fashion_mnist_train.py --epochs 5 --batch-size 64
