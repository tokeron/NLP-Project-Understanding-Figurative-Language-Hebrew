import torch
from torch.utils.data import Dataset
import pandas as pd

# Load the data and
train_df = pd.read_json('data/train.json')
test_df = pd.read_json('data/test.json')
validation_df = pd.read_json('data/validation.json')


class FLDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        item = (self.data.at(idx, 'data'), self.data.at(idx, 'tags'))
        return item

    def __len__(self):
        return len(self.data.index)
