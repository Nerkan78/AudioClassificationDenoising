import torch
from torchvision.datasets import DatasetFolder
import numpy as np
from tqdm import tqdm
import os

def read_data(folder):
    clean_files = []
    noisy_files = []

    for person_id in tqdm(os.listdir(os.path.join(folder, "clean"))):
        for file in os.listdir(os.path.join(folder, "clean", person_id)):
            data = np.load(os.path.join(folder, "clean", person_id, file))
            clean_files.append(data.mean(axis=0))

    for person_id in tqdm(os.listdir(os.path.join(folder, "noisy"))):
        for file in os.listdir(os.path.join(folder, "noisy", person_id)):
            data = np.load(os.path.join(folder, "noisy", person_id, file))
            noisy_files.append(data.mean(axis=0))

            
            
    X = np.vstack((clean_files, noisy_files))
    y = np.ones(X.shape[0])
    y[X.shape[0] // 2:] = -1

    return X, y


def npy_loader(path):
    sample = np.load(path)
    return torch.from_numpy(sample)
    
    
def get_datasets(folder):
    
    audio_dataset_clean = DatasetFolder(
        root=f"{folder}/clean",
        loader=npy_loader,
        extensions=['.npy']
    )
    audio_dataset_noisy = DatasetFolder(
        root=f"{folder}/noisy",
        loader=npy_loader,
        extensions=['.npy']
    )
    return audio_dataset_clean, audio_dataset_noisy

