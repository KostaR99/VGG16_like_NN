
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from constants import BATCH_SIZE, LEARNING_RATE, NUM_WORKERS

from cats_vs_dogs_dataset import CatsVsDogsDataset
from VGG import VGG
from training import train

if __name__ == "__main__":

    torch.cuda.empty_cache()

    transformations_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4886, 0.4555, 0.4172), (0.2526, 0.2458, 0.2487))
    ])

    transformations_val = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4836, 0.4510, 0.4152), (0.2531, 0.2469, 0.2495))
    ])

    transformations_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4869, 0.4542, 0.4151), (0.2498, 0.2421, 0.2453))
    ])

    train_d = CatsVsDogsDataset(
        dataset_path=r"D:\\cnn_vs_transformers\\data\\train",
        transformations=transformations_train)

    test_d = CatsVsDogsDataset(
        dataset_path=r"D:\\cnn_vs_transformers\\data\\test",
        transformations=transformations_test)

    val_d = CatsVsDogsDataset(
        dataset_path=r"D:\\cnn_vs_transformers\\data\\val",
        transformations=transformations_val)

    train_loader = DataLoader(
        train_d,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_d,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_d,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = VGG()

    criterion = nn.BCEWithLogitsLoss()
    opt = Adam(model.parameters(), LEARNING_RATE)

    train(model, train_loader, test_loader, criterion, opt)
    torch.cuda.empty_cache()
