import torch
from torch.utils.data import Dataset, DataLoader

import os


class MessageDataset(Dataset):
    def __init__(self, message_dir, transform=None):
        super().__init__()
        self.message_dir = message_dir
        self.messages = os.listdir(self.message_dir)
        self.transform = transform


    def __len__(self):
        return len(self.messages)

    
    def __getitem__(self, index):
        message_path = os.path.join(self.message_dir, self.messages[index])
        message = torch.load(message_path).float()
        target = message

        if self.transform:
            message = self.transform(message)

        return message, target


if __name__ == "__main__":
    """sanity check"""
    valid_dataset = MessageDataset("/root/autodl-tmp/data/valid_data")

    loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)

    datas, targets = next(iter(loader))

    print(datas.shape, targets.shape)

    print(datas[0])

