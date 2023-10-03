import torch
import torch.nn as nn
from mnist import load_train_data
from model import LogReg
from train_test import train


def main(num_epochs=2, batch_size=100, learning_rate=1e-3, save_path="model.pth"):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper-parameters
    input_size = 784  # 28x28
    num_classes = 10

    # Load data
    train_loader = load_train_data(batch_size)

    # Initiating model
    model = LogReg(input_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, criterion, optimizer, train_loader, num_epochs, input_size)

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
