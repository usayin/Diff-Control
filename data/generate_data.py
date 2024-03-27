import torch
import torch.nn as nn

import os
import argparse
from tqdm import tqdm


def make_dataset(*, 
                 seed,
                 train_folder, 
                 valid_folder, 
                 test_folder, 
                 train_size, 
                 valid_size, 
                 test_size, 
                 shape=(3, 64, 64)):
    """generate a secret bit message dataset"""
    dataset_folders = [train_folder, valid_folder, test_folder]
    dataset_sizes = [train_size, valid_size, test_size] 

    torch.manual_seed(seed)

    # check folders
    for save_folder in dataset_folders:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)


    for dataset_size, save_folder in zip(dataset_sizes, dataset_folders):
        for i in tqdm(range(dataset_size)):
            tensor = torch.randint(0, 2, shape)
            torch.save(tensor, f"{save_folder}/{i}.pt")

        print(f"successfully build dataset of size {dataset_size} and saved at {save_folder}.")  

if __name__ == "__main__":
    """Dataset for this experiment is build in the following structure

        1. Train Dataset of size 100k 
        2. Validation Dataset of size 5k
        3. Test Dataset of size 20k

        Every single element in the dataset is a tensor filled with either 0 or 1
        in the shape of (3, 64, 64) to match the input shape of Controlnet's input
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", type=str, default="data/train_data/")
    parser.add_argument("--valid_folder", type=str, default="data/valid_data/")
    parser.add_argument("--test_folder", type=str, default="data/test_data/")
    parser.add_argument("--train_size", type=int, default=100000)
    parser.add_argument("--valid_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    make_dataset(**vars(args))
