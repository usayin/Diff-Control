import torch
from torch.utils.data import Dataset, DataLoader

import os
import pandas as pd

from icecream import ic 


class MessageDataset(Dataset):
    def __init__(self, message_dir, caption_file_path=None, transform=None):
        super().__init__()
        assert caption_file_path is not None

        self.message_dir = message_dir
        self.messages = os.listdir(self.message_dir)
        self.captions = pd.read_csv(caption_file_path, header=None) 
        self.transform = transform

        if len(self.messages) != len(self.captions):
            raise ValueError("The size of messages should match the size of captions.")


    def __len__(self):
        return len(self.messages)

    
    def __getitem__(self, index):
        message_path = os.path.join(self.message_dir, self.messages[index])
        message = torch.load(message_path).float()
        caption = self.captions.iloc[index, 0]
        target = message

        if self.transform:
            message = self.transform(message)
        
        message = message.float()

        return message, target, caption


if __name__ == "__main__":
    """sanity check"""
    valid_dataset = MessageDataset("/root/autodl-tmp/data/valid_data", "/root/autodl-tmp/data/captions/valid_caption.csv")

    loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)

    datas, targets, caption = next(iter(loader))

    ic(len(caption)) 
    ic(list(caption))