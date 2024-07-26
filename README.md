# DopeVisionNet
Here is the official repo for DopeVisionNet 

This repository contains a training script for predicting molecule properties using a deep learning model. The script uses PyTorch for training and validation, with data preprocessing handled by custom dataloaders.

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- matplotlib
- tqdm
- pandas

## Setup

1. Clone the repository:

    ```bash
    git clone git@github.com:CalciumNitrade/DopeVisionNet.git
    cd DopeVisionNet
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
## Data

You can download the data for compounds from the following link: [DopeVisionNet Data](https://tamucs-my.sharepoint.com/:f:/r/personal/hasan_kurban_tamu_edu/Documents/KIL-OneDrive/Can%20Polat/DopeVisionNet/data?csf=1&web=1&e=inGKUd)

## Train Usage

The training script can be executed with various parameters. Below are the available arguments:

- `--num_epochs`: Number of training epochs (default: 100).
- `--learning_rate`: Learning rate for the optimizer (default: 0.01).
- `--batch_size`: Batch size for data loaders (default: 64).
- `--data_folder`: Path to the data folder (required).
- `--csv_file`: Path to the CSV file (required).
- `--root`: Root directory for saving results (default: `saved_model`).
- `--physicals`: List of physical properties to train (required).

### Example Command

```bash
python train.py --num_epochs 50 --learning_rate 0.001 --batch_size 32 --data_folder "/path/to/data" --csv_file "/path/to/csv" --root "/path/to/root" --physicals normalized_homo normalized_lumo
```

## Test Usage

The test script can be executed with various parameters. Below are the available arguments:

- `--data_folder`: Path to the data folder (required).
- `--csv_file`: Path to the CSV file (required).
- `--batch_size`: Batch size for data loaders (default: 350).
- `--root`: Root directory for saved models (required).
- `--physicals`: List of physical properties to evaluate (required).
- `--device`: Device to use for computation (default: `cuda`).

### Example Command

```bash
python test.py --data_folder "/path/to/data" --csv_file "/path/to/csv" --batch_size 350 --root "/path/to/saved/models" --physicals normalized_homo normalized_lumo --device cuda
```
