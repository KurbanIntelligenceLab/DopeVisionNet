import torch
import torch.nn as nn

class DopeVisionNet(nn.Module):
    def __init__(self):
        super(DopeVisionNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 16 * 16, 128)
        self.fc2 = nn.Linear(128 + 2, 64)  # Adding 2 for oxygen_num and carbon_num
        self.fc3 = nn.Linear(64 + 2, 16)   # Adding 2 for oxygen_num and carbon_num
        self.fc4 = nn.Linear(16 + 2, 1)    # Adding 2 for oxygen_num and carbon_num

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x, oxygen_num, carbon_num):
        # Ensure oxygen_num and carbon_num are reshaped correctly
        oxygen_num = oxygen_num.view(-1, 1)
        carbon_num = carbon_num.view(-1, 1)

        # Convolutional layers with ReLU activation and dropout
        x = self.activation(self.conv1(x))
        x = self.dropout(x)
        x = self.activation(self.conv2(x))
        x = self.dropout(x)
        x = self.activation(self.conv3(x))
        x = self.dropout(x)
        x = self.activation(self.conv4(x))
        x = self.dropout(x)

        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activation and dropout
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = torch.cat((x, oxygen_num, carbon_num), dim=1)  # Concatenate additional features
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = torch.cat((x, oxygen_num, carbon_num), dim=1)  # Concatenate additional features
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = torch.cat((x, oxygen_num, carbon_num), dim=1)  # Concatenate additional features
        x = self.fc4(x)
        x = self.dropout(x)

        return x