import torch
import torchvision
import torchvision.transforms as transforms


def load_train_data(batch_size):
    # MNIST dataset (images and labels)
    train_dataset = torchvision.datasets.MNIST(
        root="../../data", train=True, transform=transforms.ToTensor(), download=True
    )

    # Data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    return train_loader


def load_test_data(batch_size):
    # MNIST dataset (images and labels)
    test_dataset = torchvision.datasets.MNIST(
        root="../../data", train=False, transform=transforms.ToTensor()
    )

    # Data loader (input pipeline)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return test_loader
