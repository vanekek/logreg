import torch
from mnist import load_test_data
from model import LogReg
from train_test import test


def main(batch_size=100, save_path="model.pth", pred_path="predictions.csv"):
    # Hyper-parameters
    input_size = 784  # 28x28
    num_classes = 10

    model = LogReg(input_size, num_classes)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    test_loader = load_test_data(batch_size)

    # Test the model
    test(model, test_loader, input_size, pred_path)


if __name__ == "__main__":
    main()
