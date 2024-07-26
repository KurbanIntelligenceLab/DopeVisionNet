import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DopeVisionNetDataloader(Dataset):
    def __init__(self, data_folder, csv_file, predict_type, transform=None):
        self.data_folder = data_folder
        self.csv_file = csv_file
        self.transform = transform
        self.images = []
        self.molecule_data = {}
        self.predict_type = predict_type
        self._load_data()

    def _load_data(self):
        """Load data from the CSV file and organize it into a dictionary."""
        data_df = pd.read_csv(self.csv_file)
        for index, row in data_df.iterrows():
            dope_amount = row['dope_name']
            self.molecule_data[dope_amount] = {
                'predict_type': row[self.predict_type],
                'oxygen_amount': row['oxygen_amount'],
                'carbon_amount': row['carbon_amount']
            }

        folders = [f for f in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, f))]
        for folder in folders:
            folder_path = os.path.join(self.data_folder, folder)
            images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))]
            self.images.extend(images)

    def __len__(self):
        """Return the total number of images."""
        return len(self.images)

    def __getitem__(self, idx):
        """Return a single sample from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.images[idx]
        molecule_name = os.path.basename(os.path.dirname(image_path))
        data = self.molecule_data[molecule_name]
        prediction_value = data['predict_type']
        oxygen_number = data['oxygen_amount']
        carbon_number = data['carbon_amount']

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, prediction_value, molecule_name, oxygen_number, carbon_number