from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
# Create Training/Validation Set loader

def create_training_validation_set(path='./data', download=False, valid_size=0.1):

    validation_transformer = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    training_transformer = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    training_set = datasets.CIFAR10(
        root=path,
        train=True,
        transform=training_transformer,
        download=download
    )

    validation_set = datasets.CIFAR10(
        root=path,
        train=True,
        transform=validation_transformer,
        download=download
    )

    num_train = len(training_set)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    train_dataloader = DataLoader(training_set, batch_size=64, sampler=train_sampler)
    valid_dataloader = DataLoader(validation_set, batch_size=64, sampler=valid_sampler)

    return train_dataloader, valid_dataloader


# Create Test Set loader
def create_testing_set(path='./data', download=False):

    testing_transformer = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    testing_set = datasets.CIFAR10(
        root=path,
        train=False,
        transform=testing_transformer,
        download=download
    )

    test_dataloader = DataLoader(testing_set, batch_size=64, shuffle=True)

    return test_dataloader
