# dataset.py
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class IRDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        self.angle_map = {'eyes': 0, 'right': 1, 'front': 2, 'unknown': 3}
        self.time_map = {'sober': 0, '20mins': 1, '40mins': 2, '60mins': 3}

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('L')
        image = self.transform(image)
        label = torch.tensor(item['label'], dtype=torch.float32)
        angle = torch.tensor(self.angle_map[item['angle']])
        time = torch.tensor(self.time_map[item['time']])
        return image, angle, time, label

    def __len__(self):
        return len(self.data)
