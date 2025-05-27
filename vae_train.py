import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms
from torchvision.transforms.v2 import Grayscale
from vae_class import VAE
from torch.optim import Adam
from tqdm import tqdm
from multiprocessing import freeze_support

edge_pixels = 512

# Definiujemy transformacje (np. normalizacja, augmentacje)
# transform = transforms.Compose([
#     Grayscale(),
#     transforms.Resize((edge_pixels, edge_pixels)),
#     transforms.ToTensor()
# ])



x_dim = edge_pixels*edge_pixels*1

# Tworzymy dataset dla train i test
# train_dataset = datasets.ImageFolder(root='chest_xray/train', transform=transform)
# test_dataset = datasets.ImageFolder(root='chest_xray/test', transform=transform)

batch_size = 128
# Tworzymy DataLoader do batchowania i mieszania danych
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    # KLD = - 0.5 * torch.sum(1 + var.pow(2).log() - mean.pow(2) - var.pow(2))
    return reproduction_loss + KLD


# model = VAE(device=device).to(device)
# optimizer = Adam(model.parameters(), lr=1e-3)


def train(model, optimizer, epochs, device):
    model.train()
    for epoch in tqdm(range(epochs)):
        overall_loss = 0
        for batch_idx, (x, _) in tqdm(enumerate(train_loader)):
            x = x.view(x.shape[0], x_dim).to(device, non_blocking=True)
            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))
        torch.save(model.state_dict(), f'checkpoints/model_weights_{epoch}.pth')
        torch.cuda.empty_cache()
    return overall_loss


if __name__ == '__main__':
    freeze_support()
    # Konfiguracja datasetów i modelu
    transform = transforms.Compose([
        Grayscale(),
        transforms.Resize((edge_pixels, edge_pixels)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder(root='chest_xray/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    model = VAE(device=device).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Uruchomienie treningu
    train(model, optimizer, epochs=50, device=device)

# train(model, optimizer, epochs=30, device=device)
