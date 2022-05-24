from sklearn.svm import SVC
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from joblib import load
import argparse

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from denoising import Reconstructor
from utils import read_data, get_datasets





def eval_classifier(test_folder, model_name):
    try:
        svm_classifier = load(f'{model_name}.joblib') 
    except:
        print(f'Please provide valid name of pretrained SVM classifier')
        raise
    X_test, y_test = read_data(test_folder)
    y_test_pred = svm_classifier.predict(X_test)
    
    print(f'On test set accuracy is {accuracy_score(y_test, y_test_pred)}')
    return y_test_pred

def eval_denoising(test_folder, model_name, device):    
    model = Reconstructor(80, 40).to(device)
    try:
        model.load_state_dict(torch.load(f'{model_name}.pth'))
    except:
        print(f'Please provide valid name of model checkpoint')
        raise
    model.eval()
    
    audio_dataset_clean, audio_dataset_noisy = get_datasets(test_folder)
    test_dataloader_clean = DataLoader(audio_dataset_clean,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=2)
    test_dataloader_noisy = DataLoader(audio_dataset_noisy,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=2)
    criterion = nn.MSELoss()
    
    pbar = tqdm(zip(test_dataloader_clean, test_dataloader_noisy), total = len(test_dataloader_clean))
    total_loss = 0

    with torch.no_grad():
        for (clean_data, _), (noisy_data, _) in pbar:
            clean_data = clean_data.float().to(device)
            noisy_data = noisy_data.float().to(device)

            reconstructed = model(noisy_data)
            loss = criterion(clean_data, reconstructed)
            total_loss += loss

            pbar.set_description(f'loss : {loss.item()}')
    print(f'Average MSE loss after denoising is {total_loss.item() / len(test_dataloader_clean)}')
    
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', type=str, default=None)
    parser.add_argument('--mode', type=str, default='classification')
    parser.add_argument('--model_name', type=str, default=None)
    
    

    args = parser.parse_args()
    if args.mode == 'classification':
        eval_classifier(args.test_folder, args.model_name)
    elif args.mode == 'denoising':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        print(f'Running on {device}')
        eval_denoising(args.test_folder, args.model_name, device)
    else:
        raise NotImplementedError('Unknown format of task')
   
if __name__ == "__main__":
    main()       
    
    
        
