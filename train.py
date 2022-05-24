from sklearn.svm import SVC
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from joblib import dump
import argparse

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder

from utils import read_data, get_datasets
from denoising import Reconstructor




def train_classifier(train_folder, val_folder, model_name):
    X_train, y_train = read_data(train_folder)
    X_val, y_val = read_data(val_folder)
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)
    
    y_train_pred = svm_classifier.predict(X_train)
    y_val_pred = svm_classifier.predict(X_val)
    
    print(f'On training set accuracy is {accuracy_score(y_train, y_train_pred)}')
    print(f'On validation set accuracy is {accuracy_score(y_val, y_val_pred)}')
    
    dump(svm_classifier, f'{model_name}.joblib')


def train_denoising(train_folder, model_name, device):
    audio_dataset_clean, audio_dataset_noisy = get_datasets(train_folder)
    train_dataloader_clean = DataLoader(audio_dataset_clean,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=2)
    train_dataloader_noisy = DataLoader(audio_dataset_noisy,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=2)
    
    model = Reconstructor(80, 40).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    num_epochs = 5
    model.train()
    for epoch in (range(num_epochs)):
        pbar = tqdm(zip(train_dataloader_clean, train_dataloader_noisy), total = len(train_dataloader_clean))
        for (clean_data, _), (noisy_data, _) in pbar:
            clean_data = clean_data.float().to(device)
            noisy_data = noisy_data.float().to(device)

            optimizer.zero_grad()
            reconstructed = model(noisy_data)
            loss = criterion(clean_data, reconstructed)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'loss : {loss.item()}')
    torch.save(model.state_dict(), f'{model_name}.pth')
    
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', type=str, default=None)
    parser.add_argument('--val_folder', type=str, default=None)
    parser.add_argument('--mode', type=str, default='classification')
    parser.add_argument('--model_name', type=str, default=None)
    
    

    args = parser.parse_args()
    print(args)
    if args.mode == 'classification':
        train_classifier(args.train_folder, args.val_folder, args.model_name)
    elif args.mode == 'denoising':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        print(f'Running on {device}')
        train_denoising(args.train_folder, args.model_name, device)
    else:
        raise NotImplementedError('Unknown format of task')


if __name__ == "__main__":
    main()
    
        
