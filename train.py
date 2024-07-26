import time

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from dataloader import DopeVisionNetDataloader
from model import DopeVisionNet


def reset_weights(model):
    """Resets model weights to their initial values."""
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train(model, train_loader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        images, molecule_energies, molecule_names, oxygen_numbers, carbon_numbers = data
        inputs = images.to(device)
        labels = molecule_energies.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, oxygen_numbers.to(device), carbon_numbers.to(device))
        outputs = outputs.view(-1)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validates the model."""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, molecule_energies, molecule_names, oxygen_numbers, carbon_numbers = data
            inputs = images.to(device)
            labels = molecule_energies.to(device)

            outputs = model(inputs, oxygen_numbers.to(device), carbon_numbers.to(device))
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels.float())

            running_loss += loss.item()

    return running_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train and validate the AI model for molecule property prediction.")
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loaders.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file.')
    parser.add_argument('--root', type=str, default='saved_model', help='Root directory for saving results.')
    parser.add_argument('--physicals', nargs='+', required=True, help='List of physical properties to train.')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DopeVisionNet()

    for predict_type in args.physicals:
        for seed in range(5, 26, 5):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

            custom_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            model_result_path = f"{args.root}/{predict_type}/{seed}"

            dataset = DopeVisionNetDataloader(args.data_folder + "/train", args.csv_file, predict_type, custom_transforms)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.manual_seed(seed))
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.manual_seed(seed))

            model.to(device)
            reset_weights(model)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

            train_loss = []
            val_loss = []
            fold_duration = []
            fold_start_time = time.time()

            for epoch in tqdm(range(args.num_epochs)):
                print(f"Epoch {epoch + 1} of {args.num_epochs}")
                train_epoch_loss = train(model, train_loader, optimizer, criterion, device)
                val_epoch_loss = validate(model, val_loader, criterion, device)
                train_loss.append(train_epoch_loss)
                val_loss.append(val_epoch_loss)
                print(f"Train Loss: {train_epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")
                print(f"Train RMSE: {np.sqrt(train_epoch_loss):.4f}, Validation RMSE: {np.sqrt(val_epoch_loss):.4f}")

                if not os.path.exists(model_result_path):
                    os.makedirs(model_result_path)

                np.savetxt(f'{model_result_path}/train_loss.txt', train_loss)
                np.savetxt(f'{model_result_path}/val_loss.txt', val_loss)

                if val_epoch_loss <= min(val_loss):
                    torch.save(model.state_dict(), f"{model_result_path}/model.pth")
                    print("Saved best model weights!")

            fold_end_time = time.time()
            fold_duration.append(fold_end_time - fold_start_time)
            np.savetxt(f'{model_result_path}/train_duration.txt', fold_duration)


if __name__ == "__main__":
    main()
