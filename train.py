import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.StegoPipeline import StegoPipeline
from data.datasets import MessageDataset

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import os


def main():
    # for reproductivity
    torch.manual_seed(42)
    torch.enable_grad(True)

    # 1. prepare datasets
    train_dataset = MessageDataset(message_dir="/root/autodl-tmp/data/train_data",
                                   caption_file_path="/root/autodl-tmp/data/captions/train_caption.csv")
    valid_dataset = MessageDataset(message_dir="/root/autodl-tmp/data/valid_data",
                                   caption_file_path="/root/autodl-tmp/data/captions/valid_caption.csv")

    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=8, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, num_workers=8, shuffle=False, pin_memory=True)

    # 2. model
    stego_pipeline = StegoPipeline()

    # train mode
    checkpoint_callback = ModelCheckpoint(dirpath="/root/autodl-tmp/checkpoints/",
                                          save_top_k=3,
                                          monitor="val_loss")

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        default_root_dir="~/tf-logs/",
        val_check_interval=10000,
        callbacks=[
            checkpoint_callback,
        ],
    )

    trainer.fit(
        model=stego_pipeline,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    log_file = "best.txt"
    with open(log_file, 'w') as f:
        f.write(str(checkpoint_callback.best_model_path))


if __name__ == "__main__":
    main()