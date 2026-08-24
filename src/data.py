import torch
from torch.utils.data import Dataset, DataLoader

class StabilityDataset(Dataset):
    """Stability Dataset."""
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        item = {
            'wt_seq' : row['wt_aa_seq'],
            'mut_seq' : row['aa_seq'],
            'position': row['position']}

        if 'ddG' in self.df.columns:
            item['target'] = torch.tensor(row['ddG'], dtype = torch.float32)

        return item