import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import DopeVisionNetDataloader
from model import DopeVisionNet

# Set the criterion to L1 Loss
criterion = nn.L1Loss()

def test(model, test_loader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    running_loss = 0.0
    output = torch.zeros(0, dtype=torch.float32).to(device)

    with torch.no_grad():
        for data in test_loader:
            images, molecule_energies, molecule_names, oxygen_numbers, carbon_numbers = data
            inputs = images.to(device)
            labels = molecule_energies.to(device)

            outputs = model(inputs, oxygen_numbers.to(device), carbon_numbers.to(device))
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels.float())

            running_loss += loss.item()
            output = torch.cat((output, outputs), 0)

        output = torch.mean(output)
        labels = labels[0]
    return running_loss / len(test_loader), output.to('cpu'), labels

def main():
    parser = argparse.ArgumentParser(description="Evaluate the AI model for molecule property prediction.")
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file.')
    parser.add_argument('--batch_size', type=int, default=350, help='Batch size for data loaders.')
    parser.add_argument('--root', type=str, required=True, help='Root directory for saved models.')
    parser.add_argument('--physicals', nargs='+', required=True, help='List of physical properties to evaluate.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation (default: cuda).')

    args = parser.parse_args()

    model_vs_avg_test_loss = {}
    custom_transforms = transforms.Compose([transforms.ToTensor()])

    for predict_type in args.physicals:
        # Load test data
        test_dataset = DopeVisionNetDataloader(f"{args.data_folder}/test", args.csv_file, predict_type, custom_transforms)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        avg_loss_for_model = []
        for seed in range(5, 26, 5):
            test_loss_list = []
            chkpt_path = f"{args.root}/{predict_type}/{seed}/model.pth"

            # Initialize and load model
            model = DopeVisionNet()
            model.load_state_dict(torch.load(chkpt_path))
            model.to(args.device)

            # Evaluate model
            test_loss, output, labels = test(model, test_loader, criterion, args.device)
            test_loss_list.append(test_loss)

            avg_test_loss = np.mean(test_loss_list)
            print(f"Average test loss for seed {seed}: {avg_test_loss}")
            avg_loss_for_model.append(avg_test_loss)

        model_avg = np.mean(avg_loss_for_model)
        model_std = np.std(avg_loss_for_model)
        print(f"Average loss of all seeds for model {predict_type}: {model_avg}")

        # Calculate number of parameters
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_vs_avg_test_loss[predict_type] = [model_params, model_avg, model_std]

    print(model_vs_avg_test_loss)

if __name__ == "__main__":
    main()
