import hydra
import torch
import torch.nn as nn
from mnist import load_train_data
from model import LogReg
from omegaconf import DictConfig, OmegaConf
from train_test import train


@hydra.main(config_path="conf", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader = load_train_data(cfg["training"]["batch_size"])

    # Initiating model
    model_params = OmegaConf.to_container(cfg["model"])
    model = LogReg(**model_params).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    # Train the model
    train(
        model,
        criterion,
        optimizer,
        train_loader,
        cfg["training"]["epochs"],
        cfg["model"]["input_size"],
    )

    torch.save(model.state_dict(), cfg["training"]["save_path"])


if __name__ == "__main__":
    main()
